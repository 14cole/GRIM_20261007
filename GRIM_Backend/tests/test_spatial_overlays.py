"""Coordinate parsing, physical scaling, and real-canvas overlay interactions."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.figure import Figure
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLineEdit, QMenu

from GRIM_Backend.plotting.overlay_data import (
    axes_signature, format_coordinate, measurement_text, project_points,
    read_overlay_points, supports_overlays,
)
from test_plot_analysis_features import _WindowCase


def test_coordinate_file_formats_and_disconnected_segments(tmp_path):
    path = tmp_path/'outline.xyz'
    path.write_text('\ufeff# outline\nx,y,z\n1,2,3\n4,5,6 # second\n\n7,8,9\n',encoding='utf8')
    points = read_overlay_points(path)
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[[0, 1, 3]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.all(np.isnan(points[2]))
    for text in ('x y\n1e-3 2\n3\t4\n', 'x;y\n.001;2\n3;4\n'):
        path.write_text(text)
        np.testing.assert_allclose(read_overlay_points(path), [[.001, 2], [3, 4]])


@pytest.mark.parametrize('text', ['# empty', '1,2\n3,4,5', '1,,2', '1,NaN', '1,inf', 'x,y,z\n1,2', 'oops\n1,2'])
def test_bad_rows_are_rejected_without_silent_coordinate_loss(tmp_path, text):
    path = tmp_path/'bad.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_overlay_points(path)


def test_projection_and_mixed_axes_have_explicit_units():
    ax = Figure().add_subplot()
    ax.set_xlabel('Cross-Range (in)')
    ax.set_ylabel('Range (in)')
    np.testing.assert_allclose(project_points([[2, 4, 6]], 'YZ', ax, 'ft'), [[1.2192, 1.8288]])
    signature = axes_signature(ax)
    ax.set_xlabel('Cross-Range (m)')
    ax.set_ylabel('Range (m)')
    assert axes_signature(ax) == signature
    ax.set_xlabel('Azimuth (deg)')
    np.testing.assert_allclose(project_points([[90, 12]], 'XY', ax, 'in'), [[90, .3048]])
    assert supports_overlays(ax)
    ax.set_ylabel('Power (dB)')
    assert not supports_overlays(ax)


def test_measurements_convert_length_units_and_do_not_combine_mixed_quantities():
    ax = Figure().add_subplot()
    ax.set_xlabel('Cross-Range (in)'); ax.set_ylabel('Range (ft)')
    label, detail = measurement_text([0., 0.], [.0254, .3048], ax)
    assert label == 'Length: 12.04159 in'
    assert 'ΔX: 1.00000 in' in detail and 'ΔY: 1.00000 ft' in detail
    ax.set_xlabel('Azimuth (deg)')
    label, detail = measurement_text([90., 0.], [100., .3048], ax)
    assert 'Length' not in label
    assert 'ΔX: 10.00000 deg' in detail and 'ΔY: 1.00000 ft' in detail
    assert format_coordinate(-1e-8) == '0.00000'
    with pytest.raises(ValueError):
        measurement_text([0.,0.],[np.nan,1.],ax)


class SpatialOverlayTests(_WindowCase):
    def tearDown(self):
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        super().tearDown()

    def make_axes(self, *, unit='m', multiple=False):
        window = self.window
        window.main_tabs.setCurrentWidget(window.tab_isar)
        window.btn_auto_plot.setChecked(False)
        window.plot_figure.clear()
        axes = list(np.atleast_1d(window.plot_figure.subplots(1, 2 if multiple else 1, sharey=True)))
        for ax in axes:
            ax.imshow(np.arange(16).reshape(4, 4), extent=(-5, 5, -5, 5), origin='lower')
            ax.set_xlabel(f'Cross-Range ({unit})')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
        axes[0].set_ylabel(f'Range ({unit})')
        window.plot_ax = axes[0]
        window.plot_axes = axes if multiple else None
        window.last_plot_mode = 'isar_image'
        window.spatial_overlays.refresh()
        window.plot_canvas.draw()
        return window.spatial_overlays, axes

    def event(self, ax, x, y, kind='button_press_event', button=MouseButton.LEFT):
        px, py = ax.transData.transform([x, y])
        event = MouseEvent(kind, self.window.plot_canvas, px, py, button=button)
        getattr(self.window, {'button_press_event':'_on_plot_mouse_press',
                             'motion_notify_event':'_on_plot_mouse_move',
                             'button_release_event':'_on_plot_mouse_release'}[kind])(event)
        return event

    def test_click_draw_drag_and_navigation_are_separate(self):
        panel, (ax,) = self.make_axes()
        initial_limits = ax.get_xlim(), ax.get_ylim()
        image = ax.images[0].get_array().copy()
        panel.draw_button.setChecked(True)
        for x, y in ((-2, -1), (0, 1), (2, -1)):
            self.event(ax, x, y)
            self.event(ax, x, y, 'button_release_event')
        item, = panel.paths
        np.testing.assert_allclose(item.points, [[-2, -1], [0, 1], [2, -1]], atol=.03)
        panel.on_key(KeyEvent('key_press_event', self.window.plot_canvas, key='escape'))
        self.assertFalse(panel.draw_button.isChecked())
        self.event(ax, 0, 1)
        self.event(ax, 1, 2, 'motion_notify_event')
        self.event(ax, 1, 2, 'button_release_event')
        np.testing.assert_allclose(item.points[1], [1, 2], atol=.03)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), initial_limits)
        np.testing.assert_array_equal(ax.images[0].get_array(), image)
        previous = item.points.copy()
        self.window.btn_pan.setChecked(True)
        self.event(ax, 1, 2)
        self.event(ax, 2, 3, 'motion_notify_event')
        self.event(ax, 2, 3, 'button_release_event')
        np.testing.assert_array_equal(item.points, previous)
        self.assertNotEqual(ax.get_xlim(), initial_limits[0])
        panel.draw_button.setChecked(True)
        self.assertFalse(self.window.btn_pan.isChecked())
        self.window.btn_zoom_box.setChecked(True)
        self.assertFalse(panel.draw_button.isChecked())

    def test_loaded_xyz_style_and_unit_changes_survive_redraw(self):
        panel, (ax,) = self.make_axes(unit='in')
        item = panel.add_points([[1, 2, 3], [4, 5, 6]], plane='XZ', length_unit='in')
        np.testing.assert_allclose(item.artist.get_xydata(), [[1, 3], [4, 6]])
        panel.set_style(item, color='#00ffaa', linestyle='--', linewidth=3, show_points=False)
        old_artist = item.artist
        panel, (new_ax,) = self.make_axes(unit='m')
        self.assertIsNot(item.artist, old_artist)
        self.assertIs(item.artist.axes, new_ax)
        np.testing.assert_allclose(item.artist.get_xydata(), np.array([[1, 3], [4, 6]])*.0254)
        self.assertEqual(item.artist.get_color(), '#00ffaa')
        self.assertEqual(item.artist.get_linestyle(), '--')
        self.assertEqual(item.artist.get_linewidth(), 3)
        new_ax.set_xlabel('Frequency (GHz)')
        panel.refresh()
        self.assertIsNone(item.artist)
        self.assertIn(item, panel.paths)
        new_ax.set_xlabel('Cross-Range (m)')
        panel.refresh()
        self.assertIsNotNone(item.artist)

    def test_panel_binding_and_tab_isolation(self):
        panel, axes = self.make_axes(multiple=True)
        self.assertEqual(len(panel.axes), 2)
        panel.draw_button.setChecked(True)
        self.event(axes[1], -1, 1)
        self.event(axes[1], 1, -1)
        item, = panel.paths
        self.assertEqual(item.panel, 1)
        self.assertIs(item.artist.axes, axes[1])
        self.event(axes[0], 0, 0)
        self.assertEqual(len(item.points), 2)
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        self.assertFalse(panel.draw_button.isChecked())
        self.assertEqual(self.window.spatial_overlays.paths, [])
        self.window.main_tabs.setCurrentWidget(self.window.tab_isar)
        self.assertIs(self.window.spatial_overlays, panel)
        self.assertEqual(len(panel.paths), 1)

    def test_context_menu_routes_coordinate_editor_and_applies_values(self):
        panel, (ax,) = self.make_axes(unit='in')
        item = panel.add_points([[1, 2], [2, 3]], length_unit='in')
        self.window.plot_canvas.draw()
        px, py = ax.transData.transform([1, 2])
        ratio = self.window.plot_canvas.device_pixel_ratio
        pos = QPoint(round(px/ratio), round((self.window.plot_figure.bbox.height-py)/ratio))
        seen = []
        def choose_edit(menu, _pos):
            actions = menu.actions()
            seen.extend(a.text() for a in actions)
            next(a for a in actions if a.text() == 'Edit coordinates…').trigger()
        def fill_dialog(dialog):
            x, y = dialog.findChildren(QLineEdit)
            x.setText('2.25')
            y.setText('-3.5')
            dialog.findChild(QDialogButtonBox).accepted.emit()
            return QDialog.Accepted
        class CaptureMenu(QMenu):
            exec = choose_edit
        class EditDialog(QDialog):
            exec = fill_dialog
        with mock.patch('GRIM_Backend.ui.spatial_overlays.QMenu', CaptureMenu), mock.patch('GRIM_Backend.ui.spatial_overlays.QDialog', EditDialog):
            self.window._on_plot_context_menu(pos)
        self.assertIn('Line type', seen)
        np.testing.assert_allclose(item.points[0], np.array([2.25, -3.5])*.0254)
        with self.assertRaises(ValueError):
            panel.set_point(item, 0, float('nan'), 2)

    def test_editor_shows_five_decimals_without_rounding_unchanged_coordinates(self):
        panel, (ax,) = self.make_axes(unit='in')
        item = panel.add_points([[1.23456789123456,-2.98765432198765],[2.,3.]],length_unit='in')
        original = item.points.copy()
        class UnchangedDialog(QDialog):
            def exec(dialog):
                x,y = dialog.findChildren(QLineEdit)
                assert x.text() == '1.23457' and y.text() == '-2.98765'
                dialog.findChild(QDialogButtonBox).accepted.emit()
                return QDialog.Accepted
        with mock.patch('GRIM_Backend.ui.spatial_overlays.QDialog', UnchangedDialog):
            panel.edit_point(item,0)
        np.testing.assert_array_equal(item.points, original)
        class OneCoordinateDialog(QDialog):
            def exec(dialog):
                dialog.findChildren(QLineEdit)[0].setText('2.25')
                dialog.findChild(QDialogButtonBox).accepted.emit()
                return QDialog.Accepted
        with mock.patch('GRIM_Backend.ui.spatial_overlays.QDialog', OneCoordinateDialog):
            panel.edit_point(item,0)
        assert item.points[0,0] == 2.25*.0254
        assert item.points[0,1] == original[0,1]

    def test_click_line_measures_only_the_selected_segment_and_tracks_edits(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[-3.,-1.],[0.,3.],[3.,3.]])
        limits = ax.get_xlim(), ax.get_ylim()
        image = ax.images[0].get_array().copy()
        self.event(ax,-1.5,1.)
        assert panel.measurement == ((item,0),(item,1))
        assert 'Length: 5.00000 m' in panel.measure_label.text()
        assert len(panel.measure_artists) == 2
        panel.set_point(item,1,0.,-1.)
        assert 'Length: 3.00000 m' in panel.measure_label.text()
        assert (ax.get_xlim(),ax.get_ylim()) == limits
        np.testing.assert_array_equal(ax.images[0].get_array(),image)
        panel.remove_point(item,0)
        assert panel.measurement is None and not panel.measure_artists

    def test_measure_two_points_on_different_overlays_does_not_drag_or_draw(self):
        panel, (ax,) = self.make_axes()
        first = panel.add_points([[-2.,-1.]])
        second = panel.add_points([[1.,3.]])
        panel.draw_button.setChecked(True)
        panel.measure_button.setChecked(True)
        assert not panel.draw_button.isChecked()
        self.event(ax,-2.,-1.)
        self.event(ax,0.,0.,'motion_notify_event')
        self.event(ax,0.,0.,'button_release_event')
        np.testing.assert_array_equal(first.points,[[-2.,-1.]])
        assert panel.measure_start == (first,0)
        self.event(ax,1.,3.)
        assert panel.measurement == ((first,0),(second,0))
        assert 'Length: 5.00000 m' in panel.measure_label.text()
        panel.on_key(KeyEvent('key_press_event',self.window.plot_canvas,key='escape'))
        assert not panel.measure_button.isChecked()
        assert panel.measurement is not None
        panel.clear_measure_button.click()
        assert not panel.measure_artists and panel.measurement is None

    def test_measurement_tracks_unit_redraws_visibility_and_tab_switches(self):
        panel, (ax,) = self.make_axes()
        item = panel.add_points([[0.,0.],[.03,.04]])
        panel.measure_between((item,0),(item,1))
        assert 'Length: 0.05000 m' in panel.measure_label.text()
        panel,(ax,) = self.make_axes(unit='cm')
        assert 'Length: 5.00000 cm' in panel.measure_label.text()
        np.testing.assert_allclose(panel.measure_artists[0].get_xydata(),[[0.,0.],[3.,4.]])
        panel.set_style(item,visible=False)
        assert not panel.measure_artists and not panel.measure_label.text()
        panel.set_style(item,visible=True)
        assert 'Length: 5.00000 cm' in panel.measure_label.text()
        panel.measure_button.setChecked(True)
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        assert not panel.measure_button.isChecked()
        assert self.window.spatial_overlays.measurement is None
        self.window.main_tabs.setCurrentWidget(self.window.tab_isar)
        assert panel.measurement is not None
        self.window._clear_plot()
        assert panel.measurement is None and not panel.measure_artists

    def test_measurement_does_not_bridge_blank_lines_or_panels(self):
        panel,axes = self.make_axes(multiple=True)
        item = panel.add_points([[-4.,0.],[-3.,0.],[np.nan,np.nan],[3.,0.],[4.,0.]],ax=axes[0])
        self.event(axes[0],0.,0.)
        assert panel.measurement is None
        self.event(axes[0],-3.5,0.)
        assert panel.measurement == ((item,0),(item,1))
        panel.set_style(item,linestyle='None')
        panel.clear_measurement()
        self.event(axes[0],-3.5,0.)
        assert panel.measurement is None
        other=panel.add_points([[0.,0.]],ax=axes[1])
        panel._start_measurement(item,0)
        self.event(axes[1],0.,0.)
        assert panel.measurement is None and panel.measure_start == (item,0)
        assert 'same panel' in self.window.status.currentMessage()

    def test_navigation_and_measurement_modes_are_exclusive(self):
        panel,(ax,) = self.make_axes()
        for name in ('btn_pan','btn_zoom_box','btn_markers'):
            button=getattr(self.window,name)
            if button is None:
                continue
            button.setChecked(True)
            panel.measure_button.setChecked(True)
            assert not button.isChecked()
            button.setChecked(True)
            assert not panel.measure_button.isChecked()
            button.setChecked(False)
        item=panel.add_points([[0.,0.],[1.,1.]])
        panel._start_measurement(item,0)
        panel.hide()
        assert panel.measure_start is None and not panel.measure_button.isChecked()

    def test_right_click_measurement_actions_and_plotting_marker_mode(self):
        panel,(ax,) = self.make_axes()
        item=panel.add_points([[-2.,-1.],[1.,3.]])
        self.window.plot_canvas.draw()
        def choose_at(x,y,text):
            px,py=ax.transData.transform([x,y])
            ratio=self.window.plot_canvas.device_pixel_ratio
            pos=QPoint(round(px/ratio),round((self.window.plot_figure.bbox.height-py)/ratio))
            class CaptureMenu(QMenu):
                def exec(menu,_pos):
                    next(action for action in menu.actions() if action.text()==text).trigger()
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QMenu',CaptureMenu):
                assert panel.context_menu(pos)
        choose_at(-.5,1.,'Measure segment')
        assert 'Length: 5.00000 m' in panel.measure_label.text()
        choose_at(-2.,-1.,'Measure from this point')
        self.app.processEvents();self.window.plot_canvas.draw()
        choose_at(1.,3.,'Measure to this point')
        assert panel.measurement==((item,0),(item,1))
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        self.window.plot_figure.clear()
        ax=self.window.plot_figure.add_subplot()
        ax.set_xlabel('Cross-Range (m)');ax.set_ylabel('Range (m)')
        self.window.plot_ax=ax
        panel=self.window.spatial_overlays;panel.refresh()
        panel.measure_button.setChecked(True)
        self.window.btn_markers.setChecked(True)
        assert not panel.measure_button.isChecked()
        panel.measure_button.setChecked(True)
        assert not self.window.btn_markers.isChecked()

    def test_save_reload_preserves_segments_and_clear_removes_overlays(self):
        import tempfile
        panel, (ax,) = self.make_axes(unit='ft')
        item = panel.add_points([[1, 2], [np.nan, np.nan], [3, 4]], length_unit='ft')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'points.csv'
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QFileDialog.getSaveFileName', return_value=(str(path), 'CSV')):
                panel.save_xy()
            np.testing.assert_allclose(read_overlay_points(path), [[1, 2], [np.nan, np.nan], [3, 4]], equal_nan=True)
        panel.set_style(item, visible=False)
        self.assertIsNone(item.artist)
        panel.set_style(item, visible=True, linestyle='None', show_points=False)
        self.assertEqual(item.artist.get_marker(), 'o')
        self.window._clear_plot()
        self.assertEqual(panel.paths, [])
        self.assertFalse(panel.load_button.isEnabled())

    def test_file_load_dialog_honors_selected_plane_and_file_units(self):
        import tempfile
        panel, (ax,) = self.make_axes(unit='ft')
        class ImportDialog(QDialog):
            def exec(dialog):
                plane, unit = dialog.findChildren(QComboBox)
                plane.setCurrentText('XZ')
                unit.setCurrentText('mm')
                return QDialog.Accepted
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'outline.xyz'
            path.write_text('x,y,z\n100,200,300\n200,400,600\n')
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QDialog', ImportDialog), mock.patch(
                    'GRIM_Backend.ui.spatial_overlays.QFileDialog.getOpenFileName', return_value=(str(path),'XYZ')):
                panel.load_file()
        item, = panel.paths
        np.testing.assert_allclose(item.points, [[.1, .3], [.2, .6]])
        np.testing.assert_allclose(item.artist.get_xydata(), np.array([[.1, .3], [.2, .6]])/.3048)

    def test_failed_save_keeps_existing_file_and_cleans_temporary(self):
        import tempfile
        panel, (ax,) = self.make_axes()
        panel.add_points([[1,2],[2,3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'existing.csv'
            path.write_text('original data')
            with mock.patch('GRIM_Backend.ui.spatial_overlays.QFileDialog.getSaveFileName', return_value=(str(path),'CSV')), mock.patch(
                    'GRIM_Backend.ui.spatial_overlays.os.replace', side_effect=OSError('locked')), mock.patch(
                    'GRIM_Backend.ui.spatial_overlays.QMessageBox.warning') as warning:
                panel.save_xy()
            self.assertTrue(warning.called)
            self.assertEqual(path.read_text(),'original data')
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_export_includes_annotation_without_recording_inaccurate_replay(self):
        import tempfile
        panel, (ax,) = self.make_axes()
        item=panel.add_points([[-1,-1],[1,1]])
        panel.measure_between((item,0),(item,1))
        assert panel.measure_artists[1].get_text()=='Length: 2.82843 m'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'annotated.png'
            with mock.patch.object(self.window,'_isar_figure_is_current',return_value=True), mock.patch(
                    'GRIM_Backend.ui.dataset_actions.QFileDialog.getSaveFileName',return_value=(str(path),'PNG')), mock.patch.object(
                    self.window.python_recorder,'record_unsupported_plot') as record:
                self.window._export_plot()
            self.assertGreater(path.stat().st_size,1000)
            record.assert_called_once()
            self.assertEqual(record.call_args.args[0],'spatial_overlay')
