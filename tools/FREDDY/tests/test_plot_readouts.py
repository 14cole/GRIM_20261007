"""Hover (x, y, z) and snapped Selected (x, y, z) callouts on FREDDY maps."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from ibc.analysis_data import SweepResult
from ibc.plot import show_grid_value_on_hover
from ibc.ui import ImpedanceGui


class HoverValueTests(unittest.TestCase):
    def assert_hover_matches_drawn_cells(self, x, y, values, *, edges=False, **mesh_options):
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        mesh = axis.pcolormesh(x, y, values, **mesh_options)
        mesh.set_pickradius(0)  # exact cell hit test, the reference for the lookup
        show_grid_value_on_hover(axis, x, y, values, edges=edges)
        figure.canvas.draw()
        (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
        checked = 0
        for px, py in np.random.default_rng(7).uniform((x0, y0), (x1, y1), size=(300, 2)):
            event = MouseEvent('motion_notify_event', figure.canvas, *axis.transData.transform((px, py)))
            drawn = mesh.get_cursor_data(event)
            if drawn is None or np.size(drawn) != 1:
                continue
            xy = f'{axis.format_xdata(px)}, {axis.format_ydata(py)}'
            self.assertEqual(axis.format_coord(px, py), f'(x, y, z) = ({xy}, {float(np.ravel(drawn)[0]):.5g})')
            checked += 1
        self.assertGreater(checked, 250)
        return axis

    def test_hover_value_is_the_drawn_cell_for_uneven_and_descending_samples(self):
        values = np.arange(9.).reshape(3, 3) - 4.
        axis = self.assert_hover_matches_drawn_cells([0., 1., 10.], [6., 5., 2.], values, shading='nearest')
        # Beyond the mesh (e.g. after zooming out) only x, y remain.
        self.assertEqual(axis.format_coord(20., 3.), f'(x, y) = ({axis.format_xdata(20.)}, {axis.format_ydata(3.)})')
        self.assertEqual(axis.format_coord(None, 3.), type(axis).format_coord(axis, None, 3.))

    def test_explicit_cell_edges_like_the_failure_map(self):
        values = 100 * np.array([[0., 1., 2.], [3., 4., 8.]]) / 8
        self.assert_hover_matches_drawn_cells([8.5, 9.5, 10.5, 11.5], [-15., 15., 45.], values, edges=True)

    def test_mismatched_values_are_rejected_instead_of_mislabeling_cells(self):
        axis = Figure().add_subplot()
        with self.assertRaisesRegex(ValueError, 'do not match'):
            show_grid_value_on_hover(axis, [0., 1., 2.], [5., 6.], np.zeros((3, 2)))


class SweepSelectionReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = ImpedanceGui()
        self.panel = self.ui.analysis_panels['Off Angle']
        # Uneven frequencies [row] by angle [column]; every sample is distinct.
        self.te = -np.arange(1., 10.).reshape(3, 3)
        self.tm = self.te - 10
        upper = {'TE': {'metal_loss_db': self.te + .5}, 'TM': {'metal_loss_db': self.tm + .5}}
        self.panel.set_result(SweepResult([2., 4., 8.], [0., 30., 60.], 'Incidence angle (deg)', 'run',
                                          {'TE': {'metal_loss_db': self.te}, 'TM': {'metal_loss_db': self.tm}},
                                          upper=upper))
        self.panel.view.setCurrentText('Heatmap')

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def selection(self):
        return self.panel.selection_readout.text()

    def maps(self):
        return [axis for axis in self.panel.figure.axes if axis.get_xlabel() == 'Incidence angle (deg)']

    def test_toolbar_hover_reports_the_map_value_on_one_line(self):
        self.panel.canvas.draw()
        axis = self.maps()[0]
        event = MouseEvent('motion_notify_event', self.panel.canvas, *axis.transData.transform((29., 5.9)))
        self.panel.canvas.callbacks.process('motion_notify_event', event)
        hover = self.panel.toolbar.locLabel.text()
        self.assertTrue(hover.startswith('(x, y, z) = ('), hover)
        self.assertTrue(hover.endswith(', -5)'), hover)  # 30 deg, 4 GHz sample
        self.assertNotIn('\n', hover)

    def test_clicking_the_map_snaps_the_crosshair_and_reports_selected_xyz(self):
        self.assertEqual(self.selection(), 'Selected (x, y, z) = (0, 4, -4)')  # initial picker and mid slice
        self.panel.plot_clicked(SimpleNamespace(inaxes=self.maps()[0], button=1, xdata=58., ydata=7.))
        self.assertEqual((self.panel.selected, self.panel.frequency.value()), (2, 8.))
        self.assertEqual(self.selection(), 'Selected (x, y, z) = (60, 8, -9)')
        self.assertEqual(self.panel.selection_readout.toolTip(), self.selection())
        crosshair = self.maps()[0].lines
        self.assertEqual((list(crosshair[0].get_xdata()), list(crosshair[1].get_ydata())), ([60., 60.], [8., 8.]))
        # A typed slice between samples uses and marks the nearest computed one.
        self.panel.frequency.setValue(5.5)
        self.assertEqual(self.selection(), 'Selected (x, y, z) = (60, 4, -6)')
        self.assertEqual(list(self.maps()[0].lines[1].get_ydata()), [4., 4.])
        # z follows the displayed bound, as the colors do.
        self.panel.bound.setCurrentIndex(1)
        self.assertEqual(self.selection(), 'Selected (x, y, z) = (60, 4, -5.5)')

    def test_te_tm_maps_each_hover_their_own_grid_and_select_both(self):
        self.panel.plot_clicked(SimpleNamespace(inaxes=self.maps()[0], button=1, xdata=31., ydata=2.2))
        self.panel.view.setCurrentText('TE/TM comparison')
        self.assertEqual(self.selection(), 'Selected (x, y, z) = (30, 2, TE -2 / TM -12)')
        te_map, tm_map = self.maps()
        self.assertTrue(te_map.format_coord(59., 7.).endswith(', -9)'))
        self.assertTrue(tm_map.format_coord(59., 7.).endswith(', -19)'))
        self.assertEqual(list(tm_map.lines[1].get_ydata()), [2., 2.])

    def test_slice_view_matches_its_axes_and_curve_views_have_no_point(self):
        self.panel.selected_picker.setCurrentIndex(1)
        self.panel.view.setCurrentText('At selected frequency')
        self.assertEqual(self.selection(), 'Selected (x, y) = (30, -5)')
        self.panel.view.setCurrentText('Frequency curves')
        self.assertEqual(self.selection(), '')
        self.assertTrue(self.panel.figure.axes[0].format_coord(3., -2.).startswith('(x, y) = '))

    def test_unavailable_or_cleared_results_clear_the_selected_readout(self):
        self.panel.set_result(SweepResult([2.], [0., 30.], 'Incidence angle (deg)', 'one frequency',
                                          {'TE': {'metal_loss_db': np.array([[-3., -4.]])}}))
        self.panel.view.setCurrentText('Heatmap')
        self.assertIn('at least two', self.panel.figure.axes[0].texts[0].get_text())
        self.assertEqual(self.selection(), '')
        self.panel.view.setCurrentText('At selected frequency')
        self.assertEqual(self.selection(), 'Selected (x, y) = (0, -3)')
        self.ui._clear_analysis_results()
        self.assertEqual(self.selection(), '')

    def test_selected_readout_elides_instead_of_widening_the_toolbar(self):
        self.assertEqual(self.panel.selection_readout.minimumSizeHint().width(), 0)
        self.assertGreater(self.panel.toolbar.locLabel.minimumWidth(), 0)


class ToleranceFailureMapHoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_failure_map_hover_reports_percent_of_trials(self):
        ui = ImpedanceGui()
        self.addCleanup(ui.deleteLater)
        panel = ui.tolerance_workspace
        panel.result = {'target_db': -10., 'frequencies_ghz': [9., 10., 11.], 'angles_deg': [0., 30.],
                        'polarizations': ['te'], 'trials': 8, 'failure_counts': np.array([[[0, 1, 2], [3, 4, 8]]])}
        panel.view.setCurrentText('Failure map')
        axis = panel.figure.axes[0]
        self.assertTrue(axis.format_coord(10.2, 29.).endswith(', 50)'))
        self.assertTrue(axis.format_coord(9.2, 1.).endswith(', 0)'))


if __name__ == '__main__':
    unittest.main()
