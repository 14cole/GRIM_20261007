"""Drawn-coordinate regressions for Waterfall marginal alignment."""
from types import SimpleNamespace

import numpy as np
import pytest

from GRIM_Backend.plotting.modes import waterfall_mode
import test_image_analysis as image_helpers


@pytest.fixture
def rendered():
    owners = []

    def create(*, horizontal, vertical, colorbar, multiple=False, axes=("azimuth", "frequency")):
        owner = image_helpers.owner()
        owner.plot_figure.set_size_inches(12., 8.)
        owner.combo_plot_scale.data = "linear"
        if not multiple:
            owner._selections[owner.list_pol] = ["HH"]
            owner._selections[owner.list_elev] = [10.]
        owner.chk_colorbar.setChecked(colorbar != "off")
        owner.chk_colorbar_shared.setChecked(colorbar == "shared")
        owner.image_analysis_controls = SimpleNamespace(
            axes=lambda: axes, aggregation=lambda: "panels", linked_cuts=lambda: horizontal or vertical,
            horizontal_cuts=lambda: horizontal, vertical_cuts=lambda: vertical,
            bind=lambda controller: None)
        waterfall_mode.render(owner)
        owner.plot_canvas.draw()
        owners.append(owner)
        return owner, owner.plot_figure._grim_linked_cuts

    yield create
    for owner in owners:
        owner._release_native_plot_samples()
        owner.plot_figure.clear()


def assert_drawn_alignment(controller):
    for image, (horizontal, vertical), record in controller.panels:
        image_box = image.get_window_extent()
        if horizontal is not None:
            cut_box = horizontal.get_window_extent()
            assert cut_box.y1 < image_box.y0
            np.testing.assert_allclose([cut_box.x0, cut_box.x1], [image_box.x0, image_box.x1], atol=.05, rtol=0)
            points = np.column_stack((record.x, np.zeros_like(record.x)))
            np.testing.assert_allclose(horizontal.transData.transform(points)[:, 0],
                                       image.transData.transform(points)[:, 0], atol=.05, rtol=0)
        if vertical is not None:
            cut_box = vertical.get_window_extent()
            assert cut_box.x0 > image_box.x1
            np.testing.assert_allclose([cut_box.y0, cut_box.y1], [image_box.y0, image_box.y1], atol=.05, rtol=0)
            points = np.column_stack((np.zeros_like(record.y), record.y))
            np.testing.assert_allclose(vertical.transData.transform(points)[:, 1],
                                       image.transData.transform(points)[:, 1], atol=.05, rtol=0)


@pytest.mark.parametrize("horizontal,vertical", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("colorbar", ["off", "shared", "individual"])
@pytest.mark.parametrize("multiple", [False, True])
def test_cut_coordinates_align_after_draw_resize_and_zoom(rendered, horizontal, vertical, colorbar, multiple):
    owner, controller = rendered(horizontal=horizontal, vertical=vertical, colorbar=colorbar, multiple=multiple)
    assert_drawn_alignment(controller)
    assert len(owner.plot_figure.axes) == len(controller.panels) * (1 + horizontal + vertical) + (
        0 if colorbar == "off" else 1 if colorbar == "shared" else len(controller.panels))
    for image, (haxis, vaxis), _record in controller.panels:
        assert (haxis is not None) == horizontal
        assert (vaxis is not None) == vertical
        image.set_xlim(.25, 1.75)
        image.set_ylim(8.25, 10.75)
    owner.plot_figure.set_size_inches(15., 7.)
    owner.plot_canvas.draw()
    assert_drawn_alignment(controller)
    controller.update(0., 8.)
    owner.plot_canvas.draw()
    for image, _cut_axes, _record in controller.panels:
        np.testing.assert_allclose(image.get_xlim(), [.25, 1.75])
        np.testing.assert_allclose(image.get_ylim(), [8.25, 10.75])
    assert_drawn_alignment(controller)
    # Inverting a marginal must invert its matching image scale as well.
    for _image, (haxis, vaxis), _record in controller.panels:
        if haxis is not None:
            haxis.invert_xaxis()
        if vaxis is not None:
            vaxis.invert_yaxis()
    owner.plot_canvas.draw()
    assert_drawn_alignment(controller)


def test_each_panel_has_visible_native_cuts_and_updated_titles(rendered):
    owner, controller = rendered(horizontal=True, vertical=True, colorbar="individual", multiple=True)
    assert len({axes[0] for _, axes, _ in controller.panels}) == 4
    assert len({axes[1] for _, axes, _ in controller.panels}) == 4
    controller.update(0., 8., anchor=3)
    owner.plot_canvas.draw()
    assert_drawn_alignment(controller)
    for (_image, (horizontal, vertical), record), (hline, vline, *_cursors) in zip(controller.panels, controller.artists):
        np.testing.assert_array_equal(hline.get_ydata(), record.values[0, :])
        np.testing.assert_array_equal(vline.get_xdata(), record.values[:, 0])
        assert horizontal.get_ylim()[0] <= np.nanmin(hline.get_ydata())
        assert horizontal.get_ylim()[1] >= np.nanmax(hline.get_ydata())
        assert vertical.get_xlim()[0] <= np.nanmin(vline.get_xdata())
        assert vertical.get_xlim()[1] >= np.nanmax(vline.get_xdata())
        assert "frequency = 8" in horizontal.get_title()
        assert "azimuth = 0" in vertical.get_title()


def test_reversed_physical_axes_keep_native_values_and_pixel_alignment(rendered):
    owner, controller = rendered(horizontal=True, vertical=True, colorbar="shared",
                                 axes=("frequency", "azimuth"))
    controller.update(9., 1.)
    owner.plot_canvas.draw()
    assert_drawn_alignment(controller)
    image, (horizontal, vertical), record = controller.panels[0]
    hline, vline, *_ = controller.artists[0]
    np.testing.assert_array_equal(hline.get_xdata(), record.x)
    np.testing.assert_array_equal(hline.get_ydata(), record.values[1, :])
    np.testing.assert_array_equal(vline.get_ydata(), record.y)
    np.testing.assert_array_equal(vline.get_xdata(), record.values[:, 1])
    assert horizontal.get_xlabel() == "Frequency (GHz)"
    assert image.get_ylabel() == "Azimuth (deg)"
