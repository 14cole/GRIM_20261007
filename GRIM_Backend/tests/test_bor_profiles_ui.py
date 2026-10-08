"""BOR requests and explicit radar/body coordinates for batch drivers."""
import os, sys, unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/GHOST'))
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.bor.options import validate_options
from ghost_backend.runs.bor_setup import driver_settings


class BorProfilesUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_bor_request_driver_mapping_and_busy_controls(self):
        ghost=GhostWorkspace()
        try:
            tab=ghost.solver_tab
            original_2d=tab._capture_run_setup()
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
            tab.edit_freq_list.setText('1, 2')
            tab.edit_elev_list.setText('0, 31, 90, 180')
            tab.edit_bor_elev_list.setText('-20, 0, 20')
            tab.edit_cfie_alpha.setText('.65')
            options=validate_options(dict(factorization='compressed',angle_batch_size=17,
                compressed_storage_mib=256,compression_tile=24,tile_cache_mib=3))
            tab.bor_options_widget.set_value(options)
            recipe=tab._capture_run_setup()
            self.assertFalse(hasattr(tab,'save_run_setup_button'))
            self.assertTrue(tab.run_preflight_button.isEnabled())
            settings=driver_settings(recipe)
            self.assertEqual(settings['BOR_EXECUTION_OPTIONS'],options)
            self.assertEqual(settings['AZIMUTHS_DEG'],[0.,31.,90.,180.])
            self.assertEqual(settings['ELEVATIONS_DEG'],[-20.,0.,20.])
            self.assertEqual(settings['CFIE_ALPHA'],.65)
            self.assertEqual(settings['BODY_AXIS_EL_DEG'],0.)
            tab._set_solving_state(True)
            self.assertFalse(tab.bor_options_widget.isEnabled())
            self.assertFalse(tab.run_preflight_button.isEnabled())
            tab._set_solving_state(False)
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('2d'))
            tab.edit_freq_list.setText(', '.join(format(v, '.17g') for v in original_2d['frequencies_ghz']))
            tab.edit_elev_list.setText(', '.join(format(v, '.17g') for v in original_2d['angles_deg']))
            self.assertEqual(tab._capture_run_setup(),original_2d)
            self.assertTrue(tab.bor_options_widget.isHidden())
        finally:
            tab._set_solving_state(False)
            ghost.close()
            ghost.deleteLater()
            self.app.processEvents()


if __name__=='__main__':unittest.main()
