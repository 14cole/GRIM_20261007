"""Geometry validation and solver routing must agree on 2-D versus BoR."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ghost_backend.ui.app import GhostWorkspace

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    from PySide2.QtWidgets import QApplication


class GeometryModeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = GhostWorkspace()
        self.geometry = self.workspace.geometry_tab
        self.solver = self.workspace.solver_tab

    def tearDown(self):
        self.solver._is_solving = False
        self.solver._is_computing_density = False
        self.solver._solve_thread = None
        self.solver._density_thread = None
        self.geometry._set_dirty(False)
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def test_geometry_mode_updates_solver_controls_without_dirtying_geometry(self):
        self.assertEqual(self.geometry.geometry_mode(), "2d")
        self.geometry.set_geometry_mode("bor")
        self.assertEqual(self.solver.cmb_solver_kind.currentData(), "bor")
        self.assertFalse(self.solver.bor_options_widget.isHidden())
        self.assertFalse(self.solver.edit_bor_elev_list.isHidden())
        self.assertFalse(self.geometry.is_dirty())
        self.geometry.set_geometry_mode("2d")
        self.assertEqual(self.solver.cmb_solver_kind.currentData(), "2d")
        self.assertTrue(self.solver.bor_options_widget.isHidden())

    def test_solver_mode_updates_geometry_in_both_directions(self):
        combo = self.solver.cmb_solver_kind
        for mode in ("bor", "2d", "bor"):
            combo.setCurrentIndex(combo.findData(mode))
            self.assertEqual(self.geometry.geometry_mode(), mode)
            self.assertFalse(self.geometry.is_dirty())

    def test_geometry_selector_cannot_change_mode_during_solver_work(self):
        for task_flag in ("_is_solving", "_is_computing_density"):
            with self.subTest(task=task_flag):
                setattr(self.solver, task_flag, True)
                self.solver._apply_job_state()
                self.assertFalse(self.solver.cmb_solver_kind.isEnabled())
                self.geometry.set_geometry_mode("bor")
                self.assertEqual(self.geometry.geometry_mode(), "2d")
                self.assertEqual(self.solver.cmb_solver_kind.currentData(), "2d")
                self.assertIn("finish", self.geometry.lbl_status.text())
                setattr(self.solver, task_flag, False)
                self.solver._apply_job_state()
        self.geometry.set_geometry_mode("bor")
        self.assertEqual(self.solver.cmb_solver_kind.currentData(), "bor")

    def test_thread_teardown_still_guards_geometry_mode(self):
        class LiveThread:
            def isRunning(self):
                return True

        self.solver._solve_thread = LiveThread()
        self.geometry.set_geometry_mode("bor")
        self.assertEqual(self.geometry.geometry_mode(), "2d")
        self.assertEqual(self.solver.cmb_solver_kind.currentData(), "2d")


if __name__ == "__main__":
    unittest.main()
