"""Clearing a vehicle must remove its results, including late worker updates."""
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from GRIM_Backend.assembly.workspace import AssemblyWorkspace
from GRIM_Backend.assembly.tree import _TYPE_ROOT


class _Controls(QWidget):
    assembly_cleared = Signal()

    def __init__(self, plan):
        super().__init__()
        self.model = SimpleNamespace(prepared_plan=plan)


class ClearVehicleWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = AssemblyWorkspace()
        self.plan = SimpleNamespace(
            prepared_plan_sha256="vehicle-1",
            surface_triangles_cad_m=None,
            body_profile_rho_z_m=np.array([[0., 1.], [.3, 0.], [0., -1.]]),
            point_locations_cad_m={"fastener": np.array([[.3, 0., 0.]])},
            point_placement_ids={"fastener": ("fastener-1",)},
            line_paths_cad_m={"seam": {"line-1": np.array([[0., 0., .3], [0., .5, .3]])}},
        )
        self.controls = _Controls(self.plan)
        self.workspace.set_feature_controls(self.controls)

    def tearDown(self):
        self.workspace.response_comparison.clear_outputs()
        self.workspace.interference_inspector.clear_results()
        self._wait_for(lambda: self.workspace.response_comparison._thread is None
                       and self.workspace.interference_inspector._thread is None)
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def _wait_for(self, predicate):
        deadline = time.monotonic() + 5.
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate(), "Assembly result worker did not settle")

    @staticmethod
    def _comparison_result(generation=0):
        return {
            "_generation": generation,
            "curves": [("Body", np.array([0., 90.]), np.array([1., 2.])),
                       ("Coherent total", np.array([0., 90.]), np.array([2., 3.]))],
            "errors": [],
            "axes": (np.array([0., 90.]), np.array([0.]), np.array([10.]), np.array(["VV"])),
            "selected": (10., 0., "VV"),
        }

    def test_clear_signal_removes_current_vehicle_and_results_only(self):
        workspace = self.workspace
        tree = workspace.assembly_tree_panel.tree
        original_node = tree._make_node("Other saved response", _TYPE_ROOT, edit=False)
        workspace.add_points("other-response", [[2., 3., 4.]])
        workspace.bind_tree_item_groups(original_node, "other-response")
        workspace.load_feature_preview(self.plan)
        workspace.focus_feature_instance("point", "fastener-1")
        workspace.cmb_display_units.setCurrentIndex(workspace.cmb_display_units.findData("Feet"))
        workspace.chk_orientation_frames.setChecked(True)
        comparison = workspace.response_comparison
        comparison.show_difference.setChecked(True)
        comparison._paths = [("Body", "body.grim"), ("Coherent total", "total.grim")]
        comparison._show(self._comparison_result())
        inspector = workspace.interference_inspector
        inspector._result = {"key": ("vehicle-1", 10., 0., 0.)}
        inspector.table.setRowCount(1)
        inspector.frequency.addItem("10", 10.)
        inspector.total.setText("Previous result")
        inspector.axes.plot([0., 1.], [0., 1.])

        self.controls.model.prepared_plan = None
        self.controls.assembly_cleared.emit()

        self.assertEqual(workspace.group_ids, ("other-response",))
        self.assertIs(original_node.treeWidget(), tree)
        self.assertEqual(workspace._feature_instance_geometry, {})
        self.assertIs(workspace.viewer_tabs.currentWidget(), workspace.scene_canvas)
        self.assertEqual(workspace.scene_canvas.preview_stage, "none")
        self.assertEqual(workspace.scene_canvas.display_units, "Feet")
        self.assertTrue(workspace.chk_orientation_frames.isChecked())
        self.assertEqual(comparison._paths, [])
        self.assertIsNone(comparison._last_result)
        self.assertIsNone(comparison._difference_axes)
        self.assertEqual(len(comparison.axes.lines), 0)
        self.assertEqual(comparison.frequency.count(), 0)
        self.assertTrue(comparison.show_difference.isChecked())
        self.assertIsNone(inspector._result)
        self.assertEqual(inspector.table.rowCount(), 0)
        self.assertEqual(inspector.frequency.count(), 0)
        self.assertEqual(inspector.total.text(), "")
        self.assertEqual(len(inspector.axes.lines), 0)

    def test_replaced_or_removed_controls_cannot_clear_current_vehicle(self):
        replacement = _Controls(self.plan)
        self.workspace.set_feature_controls(replacement)
        self.workspace.load_feature_preview(self.plan)
        self.controls.assembly_cleared.emit()
        self.assertTrue(self.workspace._feature_preview_group_ids)
        replacement.assembly_cleared.emit()
        self.assertFalse(self.workspace._feature_preview_group_ids)
        self.workspace.load_feature_preview(self.plan)
        self.workspace.set_feature_controls(None)
        replacement.assembly_cleared.emit()
        self.assertTrue(self.workspace._feature_preview_group_ids)
        self.controls.deleteLater()
        replacement.deleteLater()

    def test_clear_removes_error_overlay_when_no_geometry_was_loaded(self):
        self.workspace.scene_canvas.set_feedback("error", "Previous vehicle failed")
        self.assertEqual(self.workspace.group_ids, ())

        self.controls.assembly_cleared.emit()

        self.assertEqual(self.workspace.scene_canvas.preview_state, "empty")
        self.assertNotIn("Previous vehicle failed", self.workspace.scene_canvas.feedback_text)
        self.assertIn("Nothing to preview", self.workspace.scene_canvas.feedback_text)

    def test_actual_clear_button_resets_installed_panel_and_workspace(self):
        from GRIM_Backend.assembly.panel import FeatureAssemblyPanel

        panel = FeatureAssemblyPanel(service=SimpleNamespace())
        self.workspace.set_feature_controls(panel)
        self.workspace.load_feature_preview(self.plan)
        self.workspace.response_comparison._show(self._comparison_result())
        panel._loading_recipe = True
        panel.base_picker.set_path("previous-body.grim")
        panel._loading_recipe = False
        panel.workflow_tabs.setCurrentIndex(2)

        panel.clear_all_button.click()

        self.assertEqual(panel.base_picker.path(), "")
        self.assertEqual(panel.workflow_tabs.currentIndex(), 0)
        self.assertEqual(self.workspace.group_ids, ())
        self.assertIsNone(self.workspace.response_comparison._last_result)
        self.assertIs(self.workspace.viewer_tabs.currentWidget(), self.workspace.scene_canvas)
        self.assertIsNone(self.workspace.interference_inspector.plan_provider())
        self.controls.deleteLater()

    def test_clear_cancels_comparison_queue_without_late_plot_or_tab_switch(self):
        comparison = self.workspace.response_comparison
        started, release = threading.Event(), threading.Event()
        axes = self._comparison_result()["axes"]

        def delayed_axes(_path):
            started.set()
            if not release.wait(5.):
                raise RuntimeError("test did not release comparison reader")
            return axes

        with patch("GRIM_Backend.assembly.response_comparison.response_axes", side_effect=delayed_axes):
            try:
                comparison.set_outputs("old-body.grim", None, "old-total.grim")
                self._wait_for(started.is_set)
                comparison.refresh()
                comparison._queue(None)
                self.controls.assembly_cleared.emit()
                self.assertIsNone(comparison._pending)
                self.assertTrue(comparison._cancel.is_set())
            finally:
                release.set()
            self._wait_for(lambda: comparison._thread is None)
        self.assertIsNone(comparison._last_result)
        self.assertIs(self.workspace.viewer_tabs.currentWidget(), self.workspace.scene_canvas)
        # Even a result queued before clear cannot reappear after a new token starts.
        comparison._cancel = threading.Event()
        comparison._show(self._comparison_result(generation=0))
        self.assertIsNone(comparison._last_result)
        comparison._show(self._comparison_result(generation=comparison._generation))
        self.assertIsNotNone(comparison._last_result)

    def test_clear_ignores_inspector_result_and_releases_old_vehicle_cache(self):
        inspector = self.workspace.interference_inspector
        started, release = threading.Event(), threading.Event()

        def delayed_evaluate(*_args, **_kwargs):
            started.set()
            if not release.wait(5.):
                raise RuntimeError("test did not release inspector")
            return {"error": "Old vehicle result must not appear"}

        service = SimpleNamespace(evaluate=delayed_evaluate)
        inspector._service = service
        try:
            inspector._launch(service, self.plan, (), inspector._show)
            self._wait_for(started.is_set)
            self.controls.assembly_cleared.emit()
            self.assertTrue(inspector._cancel.is_set())
            self.assertIsNone(inspector._service)
        finally:
            release.set()
        self._wait_for(lambda: inspector._thread is None)
        self.assertNotIn("Old vehicle", inspector.status.text())
        self.assertTrue(inspector.evaluate.isEnabled())
        inspector._cancel = threading.Event()
        inspector._show({"error": "Old vehicle result", "_generation": 0})
        self.assertNotIn("Old vehicle", inspector.status.text())
        inspector._show({"error": "Current vehicle result", "_generation": inspector._generation})
        self.assertEqual(inspector.status.text(), "Current vehicle result")


if __name__ == "__main__":
    unittest.main()
