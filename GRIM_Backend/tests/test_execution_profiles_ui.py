"""GHOST 2D runs expose physical choices only and use the automatic setup."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/GHOST/ghost_backend'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/GHOST'))
from PySide6.QtWidgets import QApplication
from ghost_backend.execution.options import automatic_options
from ghost_backend.ui.app import GhostWorkspace


class ExecutionProfilesUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_tools_are_collapsed_without_changing_the_request(self):
        ghost = GhostWorkspace()
        try:
            tab = ghost.solver_tab
            before = tab._capture_run_setup()
            tools = tab.tools_widget
            self.assertTrue(tools.isHidden())
            tab.btn_tools.setChecked(True)
            self.assertFalse(tools.isHidden())
            tab.btn_tools.setChecked(False)
            self.assertTrue(tools.isHidden())
            self.assertEqual(tab._capture_run_setup(), before)
            for control in (tab.btn_currents, tab.btn_solver_report):
                self.assertTrue(tools.isAncestorOf(control))
            for control in (tab.edit_geo_path, tab.cmb_solver_kind, tab.cmb_units, tab.cmb_freq_mode, tab.cmb_elev_mode,
                            tab.cmb_scatter_mode, tab.chk_mesh_certification, tab.cmb_accuracy_target,
                            tab.run_preflight_button, tab.edit_output, tab.btn_run):
                self.assertFalse(tools.isAncestorOf(control))
            for removed in ('execution_options_widget', 'geometry_preset_combo', 'cmb_solver_method', 'cmb_lu_precision',
                            'edit_quality_residual_max', 'chk_frequency_checkpoints', 'save_run_setup_button',
                            'load_run_setup_button', 'btn_advanced_settings'):
                self.assertFalse(hasattr(tab, removed), removed)
            tab.cmb_elev_mode.setCurrentIndex(1)
            tab.edit_elev_start.setText('0')
            tab.edit_elev_stop.setText('360')
            tab.edit_elev_step.setText('1')
            self.assertEqual(tab._capture_run_setup()['angles_deg'], list(range(361)))
            self.assertTrue(tab.edit_elev_list.isHidden())
            self.assertFalse(tab.elev_sweep_row.isHidden())
            self.assertIn('Azimuth', tab.lbl_angle_sweep.text())
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
            self.assertIn('Azimuth', tab.lbl_angle_sweep.text())
            self.assertFalse(tab.cmb_bor_elev_mode.isHidden())
        finally:
            ghost.close()
            ghost.deleteLater()
            self.app.processEvents()

    def test_new_windows_use_the_automatic_setup_regardless_of_launch_environment(self):
        for environment in ({}, {'GHOST_CPU_FACTORIZATION': 'dense', 'OPENBLAS_NUM_THREADS': '1'}):
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment, clear=True):
                ghost = GhostWorkspace()
                try:
                    request = ghost.solver_tab._capture_run_setup()
                    self.assertEqual((request['solver_method'], request['lu_precision']), ('auto', 'double'))
                    self.assertTrue(request['mesh_certification'])
                    self.assertEqual(request['execution_options'], automatic_options())
                finally:
                    ghost.close()
                    ghost.deleteLater()
                    self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
