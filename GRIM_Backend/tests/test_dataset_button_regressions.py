"""Known-answer and GUI regressions for the dataset/ISAR button review."""
import time
import json
import tracemalloc
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import coherent_divide, translate_phase_center
from GRIM_Backend.plotting.modes import az_vs_range_mode
from test_plot_analysis_features import _WindowCase
from test_plot_renderer_correctness import _RendererHarness


def point_grid(time_convention='exp(+jwt)', field=1j):
    return RcsGrid([0.0], [0.0], [9.0], ['HH'],
                   rcs=np.full((1, 1, 1, 1), field),
                   units={'frequency': 'GHz', 'phase_reference': 'origin',
                          'time_convention': time_convention, 'polarization_basis': 'HV'})


class NumericalButtonRegressions(unittest.TestCase):
    def test_conflicting_coherent_inputs_use_supplied_fields_and_record_advisories(self):
        left, right = point_grid(), point_grid('exp(-jwt)', -1j)
        for attested in (False, True):
            for operation, expected in (
                (lambda: left.coherent_add(right, metadata_attested=attested), 0j),
                (lambda: left.coherent_subtract(right, metadata_attested=attested), 2j),
                (lambda: coherent_divide(left, right, metadata_attested=attested), -1+0j),
            ):
                with self.subTest(attested=attested, operation=operation):
                    result = operation()
                    np.testing.assert_allclose(result.rcs, expected, atol=1e-12)
                    key = 'coherent_metadata_attestation_json' if attested else 'coherent_metadata_assumption_json'
                    self.assertIn('different time conventions', result.extra[key])
        # Equivalent spellings are compatible, and matching fields still add.
        result = left.coherent_add(point_grid('exp(+j omega t)'))
        np.testing.assert_allclose(result.rcs_power, 4.0)
        # Internal contradictions remain visible but never stop arithmetic.
        left.extra['time_convention'] = 'exp(-jwt)'
        result = left.coherent_add(point_grid(), metadata_attested=True)
        np.testing.assert_allclose(result.rcs, 2j)
        self.assertIn('different time conventions', result.extra['coherent_metadata_attestation_json'])

    def test_native_sentri_phase_center_matches_independent_direction_formula(self):
        for radians in (False, True):
            for sign in (1, -1):
                with self.subTest(radians=radians, sign=sign):
                    az = np.array([0., 30., 90.])
                    theta = np.array([0., 60., 90., 180.])
                    a, t = np.deg2rad(az)[:, None], np.deg2rad(theta)[None, :]
                    point = np.array([.4, -.25, .1])
                    projection = (np.sin(t)*np.cos(a)*point[0]
                                  + np.sin(t)*np.sin(a)*point[1] + np.cos(t)*point[2])
                    freq = np.array([9., 10.])
                    field = np.exp(sign*4j*np.pi*projection[..., None]*freq*1e9/C0)[..., None]
                    grid = RcsGrid(np.deg2rad(az) if radians else az,
                        np.deg2rad(theta) if radians else theta, freq, ['HH'], rcs=field,
                        units={'frequency': 'GHz', 'azimuth': 'rad' if radians else 'deg',
                               'elevation': 'rad' if radians else 'deg',
                               'elevation_coordinate_convention': np.array(['sentri_theta_top_zero']),
                               'time_convention': 'exp(+jwt)' if sign == 1 else 'exp(-jwt)'})
                    moved = translate_phase_center(grid, x_m=point[0], y_m=point[1], z_m=point[2])
                    np.testing.assert_allclose(moved.rcs, 1.0, atol=2e-12)
                    np.testing.assert_array_equal(moved.elevations, grid.elevations)
                    np.testing.assert_array_equal(moved.rcs_power, grid.rcs_power)
        grid.extra['sentri_elevation_convention'] = 'grim_elevation_waterline_zero_top_positive'
        with self.assertRaisesRegex(ValueError, 'contradictory elevation'):
            translate_phase_center(grid, x_m=1., y_m=0., z_m=0.)

    def test_translated_coherent_result_remains_compatible_with_the_same_reference(self):
        source = point_grid().coherent_add(point_grid())
        moved = translate_phase_center(source, x_m=.001, y_m=0., z_m=0.)
        result = moved.coherent_add(moved)
        np.testing.assert_allclose(result.rcs, 2*moved.rcs, rtol=1e-12)
        mixed = moved.coherent_add(source)
        np.testing.assert_allclose(mixed.rcs, moved.rcs + source.rcs, rtol=1e-12)
        self.assertIn('different phase references', mixed.extra['coherent_metadata_assumption_json'])

    def test_align_shrinks_first_with_bounded_memory_and_correct_complex_values(self):
        source_axes = ([0., 1.], np.arange(100.), np.arange(100.) + 1)
        a, e, f = np.meshgrid(*source_axes, indexing='ij')
        field = ((1 + a)*(2 + e)*(3 + f) + 1j*(a + e + f))[..., None]
        source = RcsGrid(*source_axes, ['HH'], rcs=field)
        target = RcsGrid(np.linspace(0, 1, 100), [49.5], [50.5], ['HH'],
                         rcs=np.ones((100, 1, 1, 1), complex))
        estimate = source._alignment_interpolation_peak_bytes(target)
        tracemalloc.start()
        try:
            result = source.align_to(target, mode='interp')
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        expected = ((1 + target.azimuths)*51.5*53.5
                    + 1j*(target.azimuths + 100.0))[:, None, None, None]
        np.testing.assert_allclose(result.rcs, expected, rtol=1e-12)
        np.testing.assert_allclose(source.rcs, field, rtol=1e-12)
        self.assertLess(peak, estimate)
        self.assertLess(peak, 8*1024**2)  # Previously about 54 MiB for this shape.

    def test_align_retains_power_only_and_local_missing_phase_samples(self):
        power = np.arange(1., 13.).reshape(2, 3, 2, 1)
        phase = np.zeros_like(power)
        phase[0, 1, 0, 0] = np.nan
        source = RcsGrid([0, 1], [0, 1, 2], [9, 10], ['HH'],
                         rcs_power=power, rcs_phase=phase)
        target = RcsGrid(np.linspace(0, 1, 5), [1], [9], ['HH'],
                         rcs_power=np.ones((5, 1, 1, 1)))
        result = source.align_to(target, mode='interp')
        np.testing.assert_allclose(result.rcs_power[:, 0, 0, 0], np.linspace(3, 9, 5))
        self.assertTrue(np.isnan(result.rcs_phase[:4]).all())
        self.assertEqual(result.rcs_phase[-1].item(), 0.)
        np.testing.assert_array_equal(source.rcs_power, power)

    def test_round_all_axes_keeps_values_precision_metadata_and_independence(self):
        power = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2, 1)
        phase = np.full_like(power, .3)
        phase[0, 0, 0, 0] = np.nan
        source = RcsGrid([.123456, 1.123456], [.234567, 1.234567], [9.345678, 10.345678],
                         ['HH'], rcs_power=power, rcs_phase=phase, history='Original',
                         extra={'phase_reference': 'origin'}, units={'frequency': 'GHz'})
        rounded = source.round_axes(3)
        sequential = source.round_azimuths(3).round_elevations(3).round_frequencies(3)
        for name in ('azimuths', 'elevations', 'frequencies', 'rcs_power', 'rcs_phase'):
            np.testing.assert_array_equal(getattr(rounded, name), getattr(sequential, name))
            self.assertFalse(np.shares_memory(getattr(rounded, name), getattr(source, name)))
        self.assertEqual(rounded.rcs_power.dtype, np.float32)
        self.assertEqual(rounded.extra['phase_reference'], 'origin')
        self.assertEqual(rounded.history, 'Original')
        source.elevations[:] = [.001, .002]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            source.round_axes(0)
        np.testing.assert_array_equal(source.rcs_power, power)


class AzimuthRangeButtonRegressions(unittest.TestCase):
    def render(self, azimuths, sign):
        frequencies = 9e9 + np.arange(256)*10e6
        distance = 17*C0/(2*256*10e6)
        field = np.exp(-sign*4j*np.pi*frequencies*distance/C0)
        field = np.array([1, 10, 1])[:, None, None, None]*field[None, None, :, None]
        grid = RcsGrid(azimuths, [0], frequencies, ['HH'], rcs=field,
                       units={'frequency': 'Hz', 'time_convention': 'exp(+jwt)' if sign == 1 else 'exp(-jwt)'})
        owner = _RendererHarness([('Point', grid)], selections={
            'azimuth': grid.azimuths, 'elevation': grid.elevations,
            'frequency': grid.frequencies, 'polarization': grid.polarizations})
        owner._selected_indices = lambda widget: set(range(len(owner._selected_values(widget))))
        owner._isar_window = lambda n: np.ones(n)
        owner.combo_isar_units = SimpleNamespace(currentText=lambda: 'm')
        az_vs_range_mode.render(owner)
        self.assertIn('updated', owner.status.message)
        return owner, distance

    def test_both_phase_conventions_locate_the_same_target_on_uniform_axes(self):
        for sign in (1, -1):
            with self.subTest(sign=sign):
                owner, expected = self.render([0, 1, 2], sign)
                image = owner.plot_ax.images[0]
                data = image.get_array()
                row, col = np.unravel_index(np.argmax(data), data.shape)
                x0, x1, y0, y1 = image.get_extent()
                self.assertAlmostEqual(x0 + (col+.5)*(x1-x0)/data.shape[1], 1.)
                self.assertAlmostEqual(y0 + (row+.5)*(y1-y0)/data.shape[0], expected)
                owner.plot_figure.clear()

    def test_irregular_azimuths_stay_at_true_coordinates_with_blank_gaps(self):
        owner, expected = self.render([0, 1, 90], 1)
        mesh = owner.plot_ax.collections[0]
        data = mesh.get_array()
        coords = mesh.get_coordinates()
        row, col = np.unravel_index(np.argmax(data), data.shape)
        self.assertAlmostEqual(np.mean(coords[row, col:col+2, 0]), 1.)
        self.assertAlmostEqual(np.mean(coords[row:row+2, col, 1]), expected)
        np.testing.assert_allclose(coords[0, :, 0], [-.5, .5, 1.5, 89.5, 90.5])
        self.assertTrue(np.ma.getmaskarray(data)[:, 2].all())
        owner.plot_figure.clear()

    def test_display_pooling_keeps_actual_edges_and_does_not_bridge_gaps(self):
        image = np.zeros((11, 5))
        image[-1, -1] = 100.
        x, y, data, reduced = az_vs_range_mode._range_display_grid(
            [0, 1, 2, 90, 91], np.arange(11.), image, azimuth_width=1., max_side=4)
        self.assertTrue(reduced)
        self.assertLessEqual(max(data.shape), 4)
        self.assertEqual(np.nanmax(data), 100.)
        np.testing.assert_allclose([x[0], x[-1], y[0], y[-1]], [-.5, 91.5, -.5, 10.5])
        gap = np.flatnonzero((x[:-1] == 2.5) & (x[1:] == 89.5))[0]
        self.assertTrue(np.isnan(data[:, gap]).all())


class DatasetButtonGuiRegressions(_WindowCase):
    def tearDown(self):
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        super().tearDown()

    def wait_job(self):
        deadline = time.monotonic() + 10
        while self.window._background_job_active() and time.monotonic() < deadline:
            self.app.processEvents()
        self.app.processEvents()
        self.assertFalse(self.window._background_job_active())

    def test_isar_has_no_hold_and_plotting_keeps_its_hold_state(self):
        self.window.btn_hold.setChecked(True)
        self.window.main_tabs.setCurrentWidget(self.window.tab_isar)
        self.assertNotIn('hold', self.window._plot_controls_by_tab['isar'])
        self.assertIsNone(self.window.btn_hold)
        self.assertTrue(self.window.chk_isar_square.isChecked())
        self.window.chk_isar_square.setChecked(False)
        self.assertFalse(self.window.chk_isar_square.isChecked())
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        self.assertTrue(self.window.btn_hold.isChecked())

    def test_phase_center_button_handles_native_theta_on_both_tabs(self):
        w = self.window
        w.table.setRowCount(0)
        native = point_grid(field=1+0j)
        native.units['elevation_coordinate_convention'] = 'sentri_theta_top_zero'
        w._add_dataset_row(native, 'Native', '', file_name='')
        for tab in (w.tab_simple_plots, w.tab_isar):
            w.main_tabs.setCurrentWidget(tab)
            self.select_rows(0)
            with mock.patch('GRIM_Backend.ui.dataset_actions.PhaseCenterDialog') as dialog:
                dialog.return_value.exec.return_value = QDialog.Accepted
                dialog.return_value.get_params.return_value = dict(
                    x_m=.001, y_m=0., z_m=0., entered=(.001, 0., 0.), unit='m')
                w.btn_phase_center.click()
                self.wait_job()
            result = w.table.item(w.table.rowCount()-1, 0).data(Qt.UserRole)
            np.testing.assert_allclose(result.rcs, 1., atol=1e-12)
            self.assertIn('created 1', w.status.currentMessage())

    def test_compatibility_warns_but_coherent_button_allows_opposite_conventions(self):
        w = self.window
        w.table.setRowCount(0)
        for grid in (point_grid(), point_grid('exp(-jwt)', -1j)):
            w._add_dataset_row(grid, grid.units['time_convention'], '', file_name='')
        for tab in (w.tab_simple_plots, w.tab_isar):
            w.main_tabs.setCurrentWidget(tab)
            self.select_rows(0, 1)
            with mock.patch('GRIM_Backend.ui.dataset_actions.DatasetCompatibilityDialog') as dialog:
                w.btn_compatibility.click()
                self.wait_job()
                report = dialog.call_args.args[0]
            self.assertIn('WARN coherent convention differences', report)
            self.assertNotIn('FAIL coherent declarations', report)
            self.assertIn('different time conventions', report)
            rows_before = w.table.rowCount()
            with mock.patch('GRIM_Backend.ui.dataset_actions.QMessageBox.question') as question:
                w.btn_coherent_add.click()
                self.wait_job()
            question.assert_not_called()
            self.assertEqual(w.table.rowCount(), rows_before + 1)
            self.assertIn('created', w.status.currentMessage())
            result = w.table.item(rows_before, 0).data(Qt.UserRole)
            np.testing.assert_allclose(result.rcs, 0j, atol=1e-12)
            record = json.loads(result.extra['coherent_metadata_assumption_json'])
            self.assertTrue(record['advisories'])
            self.assertFalse(record['user_attested'])

    def test_align_memory_preflight_blocks_before_starting_worker(self):
        w = self.window
        w.table.setRowCount(0)
        reference = RcsGrid(np.linspace(0, 1, 100), [50], [50], ['HH'],
                           rcs_power=np.ones((100, 1, 1, 1)))
        source = RcsGrid([0, 1], np.arange(100.), np.arange(100.)+1, ['HH'],
                        rcs_power=np.ones((2, 100, 100, 1)))
        for name, grid in [('Reference', reference), ('Source', source)]:
            w._add_dataset_row(grid, name, '', file_name='')
        self.select_rows(0, 1)
        with (mock.patch('GRIM_Backend.ui.dataset_actions.AlignDialog') as dialog,
              mock.patch('GRIM_Backend.ui.dataset_actions._derived_grid_memory_limit', return_value=100000),
              mock.patch.object(w, '_start_background_callable') as start):
            dialog.return_value.exec.return_value = QDialog.Accepted
            dialog.return_value.get_mode.return_value = 'interp'
            w.btn_align.click()
        start.assert_not_called()
        self.assertIn('blocked before allocation', w.status.currentMessage())


if __name__ == '__main__':
    unittest.main()
