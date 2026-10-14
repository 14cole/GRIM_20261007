"""Behavioral coverage for independently controlled, native Waterfall slices."""
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import common, waterfall_mode
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.ui.image_analysis import ImageAnalysisControls, LinkedImageCuts
import test_image_analysis as image_helpers
import test_rf_analysis_integration as rf_helpers


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def rendered(app):
    owners = []

    def create(*, horizontal=True, vertical=True, reduced=False, multiple=False, dataset=None):
        obj = image_helpers.owner(dataset)
        obj.combo_plot_scale.data = "linear"
        if not multiple:
            obj._selections[obj.list_pol] = [str(obj.active_dataset.polarizations[0])]
            obj._selections[obj.list_elev] = [float(obj.active_dataset.elevations[0])]
        controls = ImageAnalysisControls()
        controls.restore_options({"horizontal_cuts": horizontal, "vertical_cuts": vertical})
        obj.image_analysis_controls = controls
        original = common.decimate_image
        if reduced:
            with mock.patch.object(common, "decimate_image", side_effect=lambda x, y, z: original(x, y, z, max_side=2, max_cells=4)):
                waterfall_mode.render(obj)
        else:
            waterfall_mode.render(obj)
        owners.append(obj)
        return obj, controls, controls.controller

    yield create
    for obj in owners:
        obj._release_native_plot_samples()
        obj.plot_figure.clear()
        controls = obj.image_analysis_controls
        for widget in (controls.cut_bar, controls.combo_x, controls.combo_y, controls.combo_aggregation):
            widget.close(); widget.deleteLater()
        controls.deleteLater()
    app.processEvents()


def click(controller, x, y, *, panel=0, button=1):
    controller.on_click(SimpleNamespace(inaxes=controller.panels[panel][0], xdata=x, ydata=y, button=button))


@pytest.mark.parametrize("horizontal,vertical", [(True, True), (True, False), (False, True), (False, False)])
def test_each_cut_layout_contains_only_enabled_axes_and_crosshairs(rendered, horizontal, vertical):
    obj, controls, controller = rendered(horizontal=horizontal, vertical=vertical)
    image_ax, (haxis, vaxis), _record = controller.panels[0]
    assert len(obj.plot_figure.axes) == 1 + int(horizontal) + int(vertical)
    assert (haxis is not None) is horizontal
    assert (vaxis is not None) is vertical
    assert len(image_ax.lines) == int(horizontal) + int(vertical)
    hline, vline, hcursor, vcursor = controller.artists[0]
    assert (hline is not None and hcursor is not None) is horizontal
    assert (vline is not None and vcursor is not None) is vertical
    assert not controls.cut_bar.isHidden()  # The two Off switches remain reachable.
    assert controls.options()["linked_cuts"] is (horizontal or vertical)


def test_display_reduction_does_not_replace_native_weak_slice_values(rendered):
    obj, controls, controller = rendered(reduced=True)
    mesh = obj.plot_ax.collections[0]
    assert np.asarray(mesh.get_array()).size < obj.active_dataset.rcs_power[:, 0, :, 0].size
    click(controller, 1.19, 8.72)
    assert controller.position == controls.cut_position() == (1., 9.)
    hline, vline, hcursor, vcursor = controller.artists[0]
    np.testing.assert_array_equal(hline.get_xdata(), obj.active_dataset.azimuths)
    np.testing.assert_array_equal(hline.get_ydata(), obj.active_dataset.rcs_power[:, 0, 1, 0])
    np.testing.assert_array_equal(vline.get_ydata(), obj.active_dataset.frequencies)
    np.testing.assert_array_equal(vline.get_xdata(), obj.active_dataset.rcs_power[1, 0, :, 0])
    np.testing.assert_array_equal(hcursor.get_ydata(), [9., 9.])
    np.testing.assert_array_equal(vcursor.get_xdata(), [1., 1.])
    assert (controls.spin_x.value(), controls.spin_y.value()) == (1., 9.)


def test_typed_coordinate_commit_snaps_controls_lines_and_saved_position(rendered):
    _obj, controls, controller = rendered()
    for spin, text in ((controls.spin_x, "1.8"), (controls.spin_y, "8.7")):
        spin.setFocus(); spin.selectAll()
        QTest.keyClicks(spin, text); QTest.keyClick(spin, Qt.Key_Return)
    assert controller.position == (2., 9.)
    assert (controls.spin_x.value(), controls.spin_y.value()) == (2., 9.)
    assert controls.options()["cut_position"] == [2., 9.]
    hcursor, vcursor = controller.artists[0][2:]
    np.testing.assert_array_equal(hcursor.get_ydata(), [9., 9.])
    np.testing.assert_array_equal(vcursor.get_xdata(), [2., 2.])


def test_mouse_motion_and_nonleft_buttons_leave_selection_fixed(rendered):
    obj, controls, controller = rendered()
    before = controls.cut_position()
    obj.plot_canvas.draw()
    x, y = obj.plot_ax.transData.transform((0., 8.))
    motion = MouseEvent("motion_notify_event", obj.plot_canvas, x, y)
    obj.plot_canvas.callbacks.process("motion_notify_event", motion)
    assert controller.position == before
    click(controller, 0., 8., button=2)
    click(controller, 0., 8., button=3)
    assert controller.position == before
    press = MouseEvent("button_press_event", obj.plot_canvas, x, y, button=1)
    obj.plot_canvas.callbacks.process("button_press_event", press)
    assert controller.position == (0., 8.)


@pytest.mark.parametrize("mode", ["pan/zoom", "zoom rect"])
def test_active_matplotlib_toolbar_gesture_does_not_change_cuts(rendered, mode):
    obj, _controls, controller = rendered()
    before = controller.position
    with mock.patch.object(obj.plot_canvas, "toolbar", SimpleNamespace(mode=mode)):
        click(controller, 0., 8.)
    assert controller.position == before
    click(controller, 0., 8.)
    assert controller.position == (0., 8.)


@pytest.mark.parametrize("role", ["btn_pan", "btn_zoom_box", "btn_markers"])
def test_application_interaction_tools_take_priority_over_cut_selection(rendered, role):
    obj, _controls, controller = rendered()
    before = controller.position
    setattr(obj, role, SimpleNamespace(isChecked=lambda: True))
    click(controller, 0., 8.)
    assert controller.position == before
    setattr(obj, role, SimpleNamespace(isChecked=lambda: False))
    click(controller, 0., 8.)
    assert controller.position == (0., 8.)


def test_widget_lock_and_overlay_drawing_take_priority(rendered):
    obj, _controls, controller = rendered()
    before = controller.position
    lock_owner = object()
    obj.plot_canvas.widgetlock(lock_owner)
    try: click(controller, 0., 8.)
    finally: obj.plot_canvas.widgetlock.release(lock_owner)
    assert controller.position == before
    obj.spatial_overlays = SimpleNamespace(drawing=object(), drag=None)
    click(controller, 0., 8.)
    assert controller.position == before
    obj.spatial_overlays = SimpleNamespace(drawing=None, drag=object())
    click(controller, 0., 8.)
    assert controller.position == before


def test_multiple_real_panels_update_their_own_native_rows_and_columns(rendered):
    _obj, controls, controller = rendered(multiple=True)
    assert len(controller.panels) == 4
    click(controller, .95, 10.8, panel=3)
    assert controls.cut_position() == (1., 11.)
    selected_rows = []
    for (_ax, _cuts, record), (hline, vline, _hc, _vc) in zip(controller.panels, controller.artists):
        ix, iy = np.argmin(abs(record.x - 1)), np.argmin(abs(record.y - 11))
        np.testing.assert_array_equal(hline.get_ydata(), record.values[iy, :])
        np.testing.assert_array_equal(vline.get_xdata(), record.values[:, ix])
        selected_rows.append(tuple(hline.get_ydata()))
    assert len(set(selected_rows)) == 4


def test_clicked_panel_is_anchor_and_other_grids_snap_independently(app):
    figure = Figure(); canvas = FigureCanvasAgg(figure)
    ax1, ax2, horizontal, vertical = figure.subplots(1, 4)
    first = replace(image_helpers.record(), x=np.array([0., 2., 4.]), y=np.array([8., 10.]))
    second = replace(first, x=np.array([1., 3., 5.]), y=np.array([8.5, 9.5]), values=first.values + 100.)
    controller = LinkedImageCuts(canvas, [(ax1, (horizontal, vertical), first), (ax2, (horizontal, vertical), second)])
    try:
        click(controller, 2.8, 9.4, panel=1)
        assert controller.position == (3., 9.5)
        np.testing.assert_array_equal(controller.artists[0][2].get_ydata(), [10., 10.])
        np.testing.assert_array_equal(controller.artists[0][3].get_xdata(), [2., 2.])
        np.testing.assert_array_equal(controller.artists[1][2].get_ydata(), [9.5, 9.5])
        np.testing.assert_array_equal(controller.artists[1][3].get_xdata(), [3., 3.])
        assert "per panel" in horizontal.get_title()
    finally: controller.disconnect(); figure.clear()


def test_clear_releases_native_storage_and_detaches_controls_and_events(rendered):
    obj, controls, controller = rendered(reduced=True)
    old_axes = obj.plot_ax
    handle = old_axes.collections[0]._grim_native_samples._file
    connection = controller.connection
    obj._release_native_plot_samples(); obj.plot_figure.clear()
    assert handle.closed
    assert controls.controller is None and controls.cut_bar.isHidden()
    assert not controller.panels and not controller.artists
    assert connection not in obj.plot_canvas.callbacks.callbacks.get("button_press_event", {})
    controller.on_click(SimpleNamespace(inaxes=old_axes, xdata=1., ydata=9., button=1))
    assert controls.controller is None


def test_coordinate_arrows_step_native_samples_across_irregular_gaps(rendered):
    shape = (3, 1, 3, 1)
    dataset = RcsGrid([0., 1., 10.], [0.], [8., 9., 20.], ["HH"], rcs_power=np.ones(shape))
    _obj, controls, controller = rendered(dataset=dataset)
    controller.update(1., 9.)
    controls.spin_x.setFocus(); QTest.keyClick(controls.spin_x, Qt.Key_Up)
    assert controls.spin_x.value() == controller.position[0] == 10.
    QTest.keyClick(controls.spin_x, Qt.Key_Down)
    assert controls.spin_x.value() == controller.position[0] == 1.
    controls.spin_y.setFocus(); QTest.keyClick(controls.spin_y, Qt.Key_Up)
    assert controls.spin_y.value() == controller.position[1] == 20.
    QTest.keyClick(controls.spin_y, Qt.Key_Down)
    assert controls.spin_y.value() == controller.position[1] == 9.


@pytest.fixture
def real_window(app):
    case = rf_helpers.RfAnalysisIntegrationTests("test_analysis_actions_launch_from_both_tabs")
    case.setUpClass(); case.setUp(); case.load_sample()
    case.window.show(); app.processEvents()
    yield case.window
    case.window._dispose_native_plot_snapshots()
    case.tearDown(); app.processEvents()


def test_toggle_rebuild_preserves_selected_coordinates_with_real_signal_wiring(real_window, app):
    window = real_window
    window._plot_waterfall()
    controls = window.image_analysis_controls
    original = controls.controller
    click(original, 1.9, 8.9)
    assert controls.cut_position() == (2., 9.)
    for horizontal, vertical in ((True, False), (False, False), (False, True), (True, True)):
        previous = controls.controller
        controls.check_horizontal.setChecked(horizontal); controls.check_vertical.setChecked(vertical)
        app.processEvents()
        assert controls.controller is not previous
        assert not previous.panels
        assert controls.controller.position == controls.cut_position() == (2., 9.)
        assert controls.options()["horizontal_cuts"] is horizontal
        assert controls.options()["vertical_cuts"] is vertical
        for artists in controls.controller.artists:
            assert (artists[0] is not None) is horizontal
            assert (artists[1] is not None) is vertical


@pytest.mark.parametrize("horizontal,vertical", [(True, True), (True, False), (False, True), (False, False)])
def test_fit_both_retains_entire_waterfall_domain_for_each_cut_layout(real_window, horizontal, vertical):
    window = real_window
    controls = window.image_analysis_controls
    controls.restore_options({"horizontal_cuts": horizontal, "vertical_cuts": vertical})
    window._plot_waterfall()
    controls.controller.update(1., 9.)
    for ax in window.plot_axes:
        ax.set_xlim(.8, 1.2); ax.set_ylim(8.9, 9.1)
    window._plot_controls_by_tab["plotting"]["fit_both"].click()
    for ax in window.plot_axes:
        coordinates = ax.collections[0].get_coordinates()
        assert ax.get_xlim()[0] <= np.min(coordinates[..., 0])
        assert ax.get_xlim()[1] >= np.max(coordinates[..., 0])
        assert ax.get_ylim()[0] <= np.min(coordinates[..., 1])
        assert ax.get_ylim()[1] >= np.max(coordinates[..., 1])


def test_controls_hide_for_other_plot_modes_and_stay_with_plotting_tab(real_window, app):
    window = real_window
    window._plot_waterfall()
    controls = window.image_analysis_controls
    click(controls.controller, 1.9, 8.9)
    controller = controls.controller
    assert controls.cut_bar.isVisible()
    isar_index = next(index for index, key in window._tab_key_for_index.items() if key == "isar")
    plotting_index = next(index for index, key in window._tab_key_for_index.items() if key == "plotting")
    window.main_tabs.setCurrentIndex(isar_index); app.processEvents()
    assert not controls.cut_bar.isVisible()
    assert window.image_analysis_controls is None
    window.main_tabs.setCurrentIndex(plotting_index); app.processEvents()
    assert controls.cut_bar.isVisible() and controls.controller is controller
    assert controls.cut_position() == (2., 9.)
    window._plot_frequency(); app.processEvents()
    assert controls.cut_bar.isHidden()
    assert controls.controller is None


@pytest.mark.parametrize("horizontal,vertical", [(True, True), (True, False), (False, True), (False, False)])
def test_recorded_recipe_replays_independent_cuts_at_saved_native_position(horizontal, vertical):
    dataset = image_helpers.grid()
    figure = plot_datasets([("Target", dataset)], mode="waterfall", azimuths=dataset.azimuths,
                          elevations=[10.], frequencies=dataset.frequencies, polarization="HH",
                          scale="linear", show_colorbar=False,
                          heatmap_options={"horizontal_cuts": horizontal, "vertical_cuts": vertical,
                                           "cut_position": [1.8, 8.7]})
    assert len(figure.axes) == 1 + int(horizontal) + int(vertical)
    image_ax = figure.axes[0]
    assert len(image_ax.lines) == int(horizontal) + int(vertical)
    cut_axes = iter(figure.axes[1:])
    if horizontal:
        np.testing.assert_array_equal(next(cut_axes).lines[0].get_ydata(), dataset.rcs_power[:, 0, 1, 0])
    if vertical:
        np.testing.assert_array_equal(next(cut_axes).lines[0].get_xdata(), dataset.rcs_power[2, 0, :, 0])
    figure.clear()
