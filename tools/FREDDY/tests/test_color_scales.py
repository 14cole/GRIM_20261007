"""User-fixed color scales keep maps of different stacks comparable."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton

from ibc.analysis_data import SweepResult
from ibc.io import load_project_file, save_project_file
from ibc.plot import data_range, fixed_color_limits
from ibc.ui import ImpedanceGui


class ColorLimitTests(unittest.TestCase):
    def test_automatic_range_ignores_nonfinite_samples(self):
        self.assertEqual(data_range([[-3., np.nan], [np.inf, -9.]], [[-1.]]), (-9., -1.))
        self.assertIsNone(data_range([[np.nan]]))

    def test_fixed_limits_are_finite_ordered_numbers(self):
        self.assertEqual(fixed_color_limits('-40', '0'), (-40., 0.))
        for low, high, message in (('x', '0', 'numbers'), ('nan', '0', 'finite'), ('0', '0', 'greater'), (None, 1, 'numbers')):
            with self.assertRaisesRegex(ValueError, message):
                fixed_color_limits(low, high)


def stack(offset):
    """One synthetic Off Angle run: frequency rows by angle columns."""
    reflection = -np.arange(1., 10.).reshape(3, 3) * 3 - offset
    phase = np.linspace(-170., 170., 9).reshape(3, 3)
    metrics = {pol: {'metal_loss_db': reflection - shift, 'metal_phase_deg': phase}
               for pol, shift in (('TE', 0.), ('TM', 2.))}
    upper = {'TE': {'metal_loss_db': reflection + .5}}
    return SweepResult([2., 4., 8.], [0., 30., 60.], 'Incidence angle (deg)', f'stack {offset}', metrics, upper=upper)


class SweepColorScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = ImpedanceGui()
        self.panel = self.ui.analysis_panels['Off Angle']
        self.scale = self.panel.color_scale
        self.panel.set_result(stack(0.))

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def maps(self):
        return [axis for axis in self.panel.figure.axes if axis.get_xlabel() == 'Incidence angle (deg)']

    def clim(self, index=0):
        return tuple(float(v) for v in self.maps()[index].collections[0].get_clim())

    def extend(self):
        return self.maps()[0].collections[0].colorbar.extend

    def fix(self, low, high):
        self.scale.auto.setChecked(False)
        self.scale.low.setText(low)
        self.scale.high.setText(high)
        self.scale.high.editingFinished.emit()

    def test_default_scale_is_automatic_and_only_shown_for_maps(self):
        self.assertEqual((self.clim(), self.extend()), ((-27., -3.), 'neither'))
        self.assertTrue(self.scale.auto.isChecked())
        self.assertFalse(self.scale.low.isEnabled())
        self.assertEqual((self.scale.low.text(), self.scale.high.text()), ('-27', '-3'))
        self.assertEqual(self.scale.label.text(), 'Color scale · PEC reflection (dB)')
        self.assertFalse(self.scale.isHidden())
        self.panel.view.setCurrentText('Frequency curves')
        self.assertTrue(self.scale.isHidden())

    def test_fixed_scale_matches_a_second_stack_run(self):
        self.scale.auto.setChecked(False)  # locks the scale on screen, no jump
        self.assertEqual((self.clim(), self.extend()), ((-27., -3.), 'both'))
        self.fix('-40', '0')
        self.assertEqual(self.clim(), (-40., 0.))
        self.panel.set_result(stack(20.))  # the other stack's run: data -47 to -23 dB
        self.assertEqual((self.clim(), self.extend()), ((-40., 0.), 'both'))
        self.assertFalse(self.scale.auto.isChecked())
        self.panel.view.setCurrentText('TE/TM comparison')  # the same quantity shares the scale
        self.assertEqual([self.clim(0), self.clim(1)], [(-40., 0.), (-40., 0.)])
        self.panel.view.setCurrentText('Heatmap')
        self.scale.auto.setChecked(True)
        self.assertEqual((self.clim(), self.extend()), ((-47., -23.), 'neither'))
        self.assertEqual(self.scale.limits, {})

    def test_each_metric_and_tolerance_span_keeps_its_own_scale(self):
        self.fix('-40', '0')
        self.panel.metric.setCurrentText('PEC reflection phase (deg)')
        self.assertEqual(self.clim(), (-170., 170.))  # a dB scale is never applied to phase
        self.assertTrue(self.scale.auto.isChecked())
        self.assertIn('phase', self.scale.label.text())
        self.panel.metric.setCurrentText('PEC reflection (dB)')
        self.panel.bound.setCurrentIndex(3)
        self.assertTrue(self.scale.auto.isChecked())
        self.assertTrue(self.scale.label.text().endswith('tolerance span'))
        self.panel.bound.setCurrentIndex(0)
        self.assertEqual(self.clim(), (-40., 0.))

    def test_half_edited_or_invalid_limits_wait_without_changing_the_scale(self):
        self.fix('-40', '0')
        self.scale.low.setText('5')  # new Min above the old Max, before Max is typed
        self.scale.low.editingFinished.emit()
        self.assertEqual(self.clim(), (-40., 0.))
        self.assertEqual(self.scale.low.text(), '5')
        self.assertTrue(self.scale.label.text().endswith('Max must be greater than Min.'))
        self.scale.high.setText('20')
        self.scale.high.editingFinished.emit()
        self.assertEqual(self.clim(), (5., 20.))
        self.assertEqual(self.scale.label.text(), 'Color scale · PEC reflection (dB)')
        self.scale.low.setText('abc')
        self.scale.low.editingFinished.emit()
        self.assertIn('numbers', self.scale.label.text())
        self.assertEqual(self.clim(), (5., 20.))
        self.panel.frequency.setValue(8.)  # any redraw shows the applied limits again
        self.assertEqual((self.scale.low.text(), self.scale.high.text()), ('5', '20'))

    def test_fixed_scales_round_trip_through_projects(self):
        self.fix('-40', '0')
        state = self.ui._collect_project_state()
        self.assertEqual(state['color_scales'], {'Off Angle': {'metal_loss_db': [-40., 0.]}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'project.json'
            save_project_file(path, state)
            loaded = load_project_file(path)
        other = ImpedanceGui()
        self.addCleanup(other.deleteLater)
        other._apply_project_state(loaded)
        self.assertEqual(other.analysis_panels['Off Angle'].color_scale.limits, {'metal_loss_db': (-40., 0.)})
        other._mark_project_clean()
        other.analysis_panels['Off Angle'].color_scale.limits.clear()
        self.assertTrue(other.is_dirty())

    def test_older_projects_keep_session_scales_and_bad_ones_are_rejected(self):
        self.fix('-40', '0')
        older = self.ui._collect_project_state()
        older.pop('color_scales')  # saved before color scales existed, or with none fixed
        self.ui._apply_project_state(older)
        self.assertEqual(self.scale.limits, {'metal_loss_db': (-40., 0.)})
        for bad in ({'Off Angle': {'metal_loss_db': [0, -40]}}, {'Off Angle': [-40, 0]}, [], {'Off Angle': {'metal_loss_db': [1]}}):
            with self.assertRaises(ValueError):
                self.ui._apply_project_state(dict(older, color_scales=bad))
        self.assertEqual(self.scale.limits, {'metal_loss_db': (-40., 0.)})
        self.ui._apply_project_state(dict(older, color_scales={'Thickness': {'metal_loss_db': [-30, 0]}}))
        self.assertEqual(self.scale.limits, {})  # a project's saved scales replace the session's
        self.assertEqual(self.ui.analysis_panels['Thickness'].color_scale.limits, {'metal_loss_db': (-30., 0.)})


class MixColorScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_material_mix_map_follows_its_auto_min_max_controls(self):
        ui = ImpedanceGui()
        self.addCleanup(ui.deleteLater)
        ui._select_mode(ui._mode_labels.index('Material Mix'))
        performance = {'label': 'PEC reflection', 'unit': 'dB', 'target': -10., 'direction': 'at_most', 'gap': 1.,
                       'freqs': [8., 10., 12.], 'angles': [0., 30.], 'grid': [[-5., -8.], [-12., -15.], [-9., -3.]]}
        ui.mix_preview = {'freqs': performance['freqs'], 'performance': performance}
        ui._update_plot()
        mesh = lambda: ui.ax_heatmap.collections[0]
        self.assertEqual(tuple(mesh().get_clim()), (-15., -3.))
        auto = next(box for box in ui.findChildren(QCheckBox) if box.text() == 'Auto color scale')
        auto.click()  # prefills the scale on screen instead of failing on empty limits
        self.assertEqual((ui.cbar_min_var.get(), ui.cbar_max_var.get()), ('-15', '-3'))
        self.assertEqual(mesh().colorbar.extend, 'both')
        ui.cbar_min_var.set('-20')
        ui.cbar_max_var.set('0')
        ui._update_plot()
        self.assertEqual(tuple(mesh().get_clim()), (-20., 0.))
        ui.cbar_max_var.set('-30')
        with mock.patch('ibc.ui.messagebox.showerror') as error:
            ui._update_plot()
        error.assert_called_once()
        self.assertEqual(tuple(mesh().get_clim()), (-15., -3.))  # invalid limits fall back to automatic

    def test_only_material_mix_shows_the_plot_strip_and_it_holds_what_applies(self):
        ui = ImpedanceGui()
        self.addCleanup(ui.deleteLater)
        ui.resize(1250, 850)
        ui.show()
        for index, label in enumerate(ui._mode_labels):
            ui._select_mode(index)
            self.app.processEvents()
            self.assertEqual(ui.results_pane.isVisible(), label == 'Material Mix', label)
        strip = next(box for box in ui.results_pane.findChildren(QGroupBox) if box.title() == 'Plot Controls')
        texts = [widget.text() for kind in (QCheckBox, QLabel, QPushButton) for widget in strip.findChildren(kind)]
        self.assertEqual(sorted(texts), ['Auto color scale', 'Max', 'Min', 'Save Plot'])
        self.assertEqual(len(strip.findChildren(QLineEdit)), 2)
        self.assertEqual(strip.findChildren(QComboBox), [])
        ui._select_mode(ui._mode_labels.index('Material Mix'))
        save = next(button for button in strip.findChildren(QPushButton) if button.text() == 'Save Plot')
        self.assertTrue(save.isVisible())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mix.png'
            with mock.patch('ibc.ui.filedialog.asksaveasfilename', return_value=str(path)), \
                 mock.patch('ibc.ui.messagebox.showinfo'), \
                 mock.patch.object(ui.fig, 'savefig', wraps=ui.fig.savefig) as savefig:
                save.click()
            savefig.assert_called_once()  # the Material Mix figure
            self.assertGreater(path.stat().st_size, 0)

    def test_projects_with_retired_heatmap_keys_load_and_keep_mix_limits(self):
        ui = ImpedanceGui()
        self.addCleanup(ui.deleteLater)
        retired = {'heatmap_metric', 'uncertainty_view', 'slice_angle', 'slice_freq'}
        older = ui._collect_project_state()
        self.assertTrue(retired.isdisjoint(older['controls']))
        # Written by a FREDDY that still had the Off Angle/Thickness heatmap strip.
        older['controls'].update(heatmap_metric='Metal backed loss (dB)', uncertainty_view='Span (max-min)',
                                 slice_angle='30', slice_freq='10', cbar_auto=False, cbar_min='-30', cbar_max='0')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'older.json'
            # Outputs inside the project folder keep the load free of portability warnings.
            older['controls'].update({key: str(Path(folder) / key) for key in
                                      ('output', 'ibc_batch_output_dir', 'angle_output', 'thk_output')})
            save_project_file(path, older)
            ui._apply_project_state(load_project_file(path))
            self.assertEqual((ui.cbar_auto_var.get(), ui.cbar_min_var.get(), ui.cbar_max_var.get()), (False, '-30', '0'))
            saved = ui._collect_project_state()
            self.assertTrue(retired.isdisjoint(saved['controls']))
            save_project_file(path, saved)
            other = ImpedanceGui()
            self.addCleanup(other.deleteLater)
            other._apply_project_state(load_project_file(path))
        self.assertEqual((other.cbar_auto_var.get(), other.cbar_min_var.get(), other.cbar_max_var.get()),
                         (False, '-30', '0'))


if __name__ == '__main__':
    unittest.main()
