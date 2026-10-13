"""Whole-outline transforms, floating controls, and reversible previews."""
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QMenu

from GRIM_Backend.plotting.overlay_data import read_overlay_points, transform_points
from test_plot_analysis_features import _WindowCase
import test_spatial_overlays as spatial_tests


def test_transform_order_pivot_separators_and_source_precision():
    original = np.array([[3., 4.], [np.nan, np.nan], [1., 2.]])
    frozen = original.copy()
    # Around (1, 2): (2, 2) -> (-2, 2) -> (-2, -2), then move (10, 20).
    changed = transform_points(original, mirror_x=True, angle_deg=90,
                               pivot=(1, 2), offset=(10, 20))
    np.testing.assert_allclose(changed, [[9, 20], [np.nan, np.nan], [11, 22]], atol=1e-14)
    np.testing.assert_array_equal(original, frozen)
    values = np.array([[1.123456789123456, 1e-25]])
    np.testing.assert_array_equal(transform_points(values, pivot=(1e100, 1e100)), values)
    mirrored = transform_points(values, mirror_x=True, pivot=(0, 1e100))
    np.testing.assert_array_equal(mirrored, values * [-1, 1])
    np.testing.assert_array_equal(transform_points(values, angle_deg=360), values)


@pytest.mark.parametrize('options', [dict(angle_deg=np.inf), dict(offset=(np.nan, 0)),
                                    dict(pivot=(0, np.inf)), dict(offset=(1, 2, 3))])
def test_bad_transform_values_are_rejected(options):
    with pytest.raises(ValueError):
        transform_points([[1, 2]], **options)


def test_overflow_and_partial_separator_cannot_corrupt_overlay():
    for points, options in (([[1e308, 0]], dict(offset=(1e308, 0))),
                            ([[1, np.nan]], {}), ([[np.inf, 1]], {})):
        with pytest.raises(ValueError):
            transform_points(points, **options)


class OverlayTransformTests(_WindowCase):
    make_axes = spatial_tests.SpatialOverlayTests.make_axes
    event = spatial_tests.SpatialOverlayTests.event
    def tearDown(self):
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        super().tearDown()

    def test_controls_and_editor_are_nonmodal_and_do_not_resize_plot(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[-1, -2], [1, 2]])
        self.app.processEvents()
        self.window.plot_canvas.draw()
        size = self.window.plot_canvas.size()
        bounds = ax.get_position().bounds
        limits = ax.get_xlim(), ax.get_ylim()
        pixels = ax.images[0].get_array().copy()
        clim = ax.images[0].get_clim()
        panel.toggle_button.click()
        self.app.processEvents()
        self.assertTrue(panel.isWindow())
        self.assertFalse(panel.isModal())
        self.assertTrue(panel.isVisible())
        panel.edit_button.click()
        dialog = panel.transform_dialog
        self.assertFalse(dialog.isModal())
        self.assertFalse(panel.save_button.isEnabled())
        dialog.rotation.setValue(90)
        dialog.translation[0].setText('1')
        np.testing.assert_allclose(item.points, [[3, -1], [-1, 1]], atol=1e-14)
        self.app.processEvents()
        self.window.plot_canvas.draw()
        self.assertEqual(self.window.plot_canvas.size(), size)
        np.testing.assert_allclose(ax.get_position().bounds, bounds, atol=1e-10)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        self.assertEqual(ax.images[0].get_clim(), clim)
        np.testing.assert_array_equal(ax.images[0].get_array(), pixels)
        panel.close()
        self.assertFalse(panel.toggle_button.isChecked())
        self.assertIsNone(panel.transform_dialog)
        self.assertTrue(panel.save_button.isEnabled())
        np.testing.assert_array_equal(item.points, [[-1, -2], [1, 2]])
        panel.toggle_button.click()
        self.assertTrue(panel.isVisible())
        self.assertIsNotNone(item.artist)

    def test_live_preview_is_not_cumulative_and_reset_cancel_restore_exact_points(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[1.12345678901234, 2], [np.nan, np.nan], [4, 6]])
        original = item.points.copy()
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.translation[0].setText('2')
        dialog.translation[0].setText('3')
        np.testing.assert_allclose(item.points, original + [3, 0])
        dialog.buttons.button(QDialogButtonBox.Reset).click()
        np.testing.assert_array_equal(item.points, original)
        dialog.mirror_x.setChecked(True)
        dialog.rotation.setValue(-60)
        dialog.close()
        np.testing.assert_array_equal(item.points, original)
        self.assertIsNone(panel.transform_dialog)

    def test_apply_establishes_new_baseline_and_escape_cancels_only_next_preview(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 0], [2, 4]])
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.translation[0].setText('2')
        dialog.buttons.button(QDialogButtonBox.Apply).click()
        applied = item.points.copy()
        np.testing.assert_array_equal(applied, [[2, 0], [4, 4]])
        self.assertEqual(dialog.translation[0].text(), '0')
        self.assertTrue(dialog.isVisible())
        dialog.translation[1].setText('9')
        QTest.keyClick(dialog, Qt.Key_Escape)
        np.testing.assert_array_equal(item.points, applied)
        self.assertIsNone(panel.transform_dialog)

    def test_rotation_uses_physical_distance_with_different_axis_units(self):
        panel, (ax,) = self.make_axes(unit='in')
        ax.set_ylabel('Range (ft)')
        panel.refresh()
        item = panel.add_points([[12, 0], [0, 12]], length_unit='in')
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.pivot_mode.setCurrentIndex(1)
        dialog.rotation.setValue(90)
        dialog.translation[0].setText('2')
        dialog.translation[1].setText('1')
        dialog.buttons.button(QDialogButtonBox.Ok).click()
        np.testing.assert_allclose(item.points, [[.0508, .6096], [-.254, .3048]], atol=1e-14)
        np.testing.assert_allclose(item.artist.get_xydata(), [[2, 2], [-10, 1]], atol=1e-13)

    def test_custom_pivot_preserves_unedited_center_precision(self):
        panel, (ax,) = self.make_axes(unit='in')
        item = panel.add_points([[.12345678912345, .98765432198765], [2, 3]], length_unit='in')
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.rotation.setValue(30)
        center_rotation = item.points.copy()
        self.assertEqual(dialog.pivot_fields[0].text(), '1.06173')
        dialog.pivot_mode.setCurrentIndex(2)
        np.testing.assert_array_equal(item.points, center_rotation)
        dialog.accept()

    def test_custom_pivot_both_mirrors_and_mixed_axis_translation(self):
        panel, (ax,) = self.make_axes(unit='ft')
        ax.set_xlabel('Azimuth (deg)')
        panel.refresh()
        item = panel.add_points([[10, 2], [20, 4]], length_unit='ft')
        panel.edit_overlay()
        dialog = panel.transform_dialog
        self.assertFalse(dialog.rotation.isEnabled())
        dialog.pivot_mode.setCurrentIndex(2)
        dialog.pivot_fields[0].setText('5')
        dialog.pivot_fields[1].setText('1')
        dialog.mirror_x.setChecked(True)
        dialog.mirror_y.setChecked(True)
        dialog.translation[0].setText('30')
        dialog.translation[1].setText('2')
        dialog.accept()
        np.testing.assert_allclose(item.points, [[30, .6096], [20, 0]], atol=1e-14)

    def test_invalid_preview_keeps_last_valid_value_and_blocks_apply(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 1], [2, 3]])
        original = item.points.copy()
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.translation[0].setText('2')
        valid = item.points.copy()
        for value in ('-', 'nan', 'inf', '1e999'):
            dialog.translation[0].setText(value)
            self.assertFalse(dialog.buttons.button(QDialogButtonBox.Apply).isEnabled())
            self.assertFalse(dialog.buttons.button(QDialogButtonBox.Ok).isEnabled())
            self.assertTrue(dialog.error_label.text())
            np.testing.assert_array_equal(item.points, valid)
        dialog.reject()
        np.testing.assert_array_equal(item.points, original)

    def test_preview_updates_measurement_without_editing_other_overlays_or_dragging(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 0], [1, 1]])
        other = panel.add_points([[2, 2], [3, 3]])
        panel.measure_between((item, 0), (other, 0))
        detail = panel.measure_label.text()
        panel._activate(item)
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.translation[0].setText('1')
        self.assertNotEqual(panel.measure_label.text(), detail)
        np.testing.assert_array_equal(other.points, [[2, 2], [3, 3]])
        self.event(ax, 1, 0)
        self.assertIsNone(panel.drag)
        dialog.reject()
        self.assertEqual(panel.measure_label.text(), detail)

    def test_replot_clear_and_tab_switch_cancel_pending_transform(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 0], [1, 1]])
        original = item.points.copy()
        for action in ('replot', 'tab', 'clear'):
            panel.edit_overlay()
            panel.transform_dialog.translation[0].setText('20')
            if action == 'replot':
                self.make_axes(unit='in')
                np.testing.assert_allclose(item.artist.get_xydata(), original / .0254)
            elif action == 'tab':
                self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
                self.assertFalse(panel.isVisible())
                self.assertFalse(panel.toggle_button.isChecked())
                self.window.main_tabs.setCurrentWidget(self.window.tab_isar)
            else:
                panel.clear()
                self.assertEqual(panel.paths, [])
            self.assertIsNone(panel.transform_dialog)
            np.testing.assert_array_equal(item.points, original)

    def test_right_click_can_open_transform_editor_without_docking_controls(self):
        panel, (ax,) = self.make_axes()
        panel.add_points([[1, 2], [2, 3]])
        self.window.plot_canvas.draw()
        px, py = ax.transData.transform([1, 2])
        ratio = self.window.plot_canvas.device_pixel_ratio
        pos = QPoint(round(px / ratio), round((self.window.plot_figure.bbox.height - py) / ratio))
        class TransformMenu(QMenu):
            def exec(menu, _pos):
                next(action for action in menu.actions() if action.text() == 'Edit overlay…').trigger()
        with mock.patch('GRIM_Backend.ui.spatial_overlays.QMenu', TransformMenu):
            self.window._on_plot_context_menu(pos)
        self.assertIsNotNone(panel.transform_dialog)
        self.assertTrue(panel.isWindow())

    def test_plot_navigation_remains_available_during_transform_preview(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 0], [1, 1]])
        panel.edit_overlay()
        dialog = panel.transform_dialog
        dialog.translation[0].setText('1')
        preview = item.points.copy()
        limits = ax.get_xlim()
        self.window.btn_pan.setChecked(True)
        self.event(ax, 0, 0)
        self.event(ax, 1, 1, 'motion_notify_event')
        self.event(ax, 1, 1, 'button_release_event')
        self.assertNotEqual(ax.get_xlim(), limits)
        np.testing.assert_array_equal(item.points, preview)
        self.assertIs(panel.transform_dialog, dialog)
        dialog.reject()

    def test_leaving_plot_tabs_closes_popups_and_reverts_preview(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0, 0], [1, 1]])
        panel.edit_overlay()
        panel.transform_dialog.translation[0].setText('4')
        self.window.main_tabs.setCurrentWidget(self.window.ghost_integration)
        self.assertFalse(panel.isVisible())
        self.assertIsNone(panel.transform_dialog)
        self.assertFalse(panel.toggle_button.isChecked())
        np.testing.assert_array_equal(item.points, [[0, 0], [1, 1]])

    def test_load_transform_save_reload_preserves_gaps_and_leaves_source_unchanged(self):
        panel, (ax,) = self.make_axes(unit='in')
        class ImportDialog(QDialog):
            def exec(dialog):
                plane, unit = dialog.findChildren(QComboBox)
                plane.setCurrentText('XZ')
                unit.setCurrentText('in')
                return QDialog.Accepted
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'original.xyz', Path(directory) / 'edited.csv'
            content = 'x,y,z\n0,99,0\n2,99,0\n\n0,99,2\n'
            source.write_text(content)
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QFileDialog.getOpenFileName', return_value=(str(source), '')), mock.patch('GRIM_Backend.ui.spatial_overlays.QDialog', ImportDialog):
                panel.load_file()
            item, = panel.paths
            panel.edit_overlay()
            dialog = panel.transform_dialog
            dialog.pivot_mode.setCurrentIndex(1)
            dialog.rotation.setValue(90)
            dialog.translation[0].setText('1')
            dialog.accept()
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QFileDialog.getSaveFileName', return_value=(str(target), 'CSV')):
                panel.save_xy()
            self.assertEqual(source.read_text(), content)
            saved = read_overlay_points(target)
            np.testing.assert_allclose(saved, [[1, 0], [1, 2], [np.nan, np.nan], [-1, 0]], atol=1e-14)
            again = panel.add_points(saved, length_unit='in')
            np.testing.assert_allclose(again.points, item.points, atol=1e-14)
