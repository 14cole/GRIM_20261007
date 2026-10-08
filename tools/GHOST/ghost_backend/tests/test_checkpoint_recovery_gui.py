"""Partial solve results are visible and never automatically replace exports."""
import os
import unittest
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from test_pipeline_performance import small_result


class CheckpointRecoveryGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_partial_result_labels_unfinished_grid_and_keeps_manual_export(self):
        workspace = GhostWorkspace()
        try:
            tab = workspace.solver_tab
            tab.chk_export_after_solve.setChecked(True)
            result = small_result(.6)
            result['metadata'].update(partial_result=True, requested_frequency_count=3,
                remaining_frequencies_ghz=[.8, 1.], frequency_checkpoints=dict(completed=1,
                    persisted=0, reused=0, directory='', write_warnings=['disk full']))
            for row in result['samples']:
                row['rcs_linear'] = 1.
            with mock.patch.object(tab, '_export_result_files') as export:
                tab._on_solver_finished(result, '')
            export.assert_not_called()
            self.assertIs(tab.last_result, result)
            self.assertIn('PARTIAL RESULT (1/3 frequencies)', tab.lbl_status.text())
            self.assertIn('Manual export', tab.lbl_status.text())
            self.assertEqual(tab.progress.value(), 33)
        finally:
            workspace.close()


if __name__ == '__main__':
    unittest.main()
