"""Thin-gap measurement, material isolation, and detail preview behavior."""
import copy
from contextlib import contextmanager
import threading
import time
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.ui.geometry_inspection import _GapWorker

THIN = """Title: Thin gaps
Segment: lower 2
properties: 2 10 0 0 0
0 0 2 0
Segment: upper 2
properties: 2 10 0 0 0
0 0.01 2 0.01
IBCS_Resistances:
Dielectrics:
"""
CROSSED = """Title: Crossed boundaries
Segment: rising 2
properties: 2 10 0 0 0
0 0 2 2
Segment: falling 2
properties: 2 10 0 0 0
0 2 2 0
IBCS_Resistances:
Dielectrics:
"""
NESTED = """Title: Coating and separate dielectric
Segment: coating 3
properties: 3 10 0 1 0
-2 -2 -2 2
-2 2 2 2
2 2 2 -2
2 -2 -2 -2
Segment: core 4
properties: 4 10 0 1 0
-1 -1 -1 1
-1 1 1 1
1 1 1 -1
1 -1 -1 -1
Segment: second_material 3
properties: 3 10 0 2 0
4 -1 4 1
4 1 6 1
6 1 6 -1
6 -1 4 -1
IBCS_Resistances:
Dielectrics:
1 3 0 1 0
2 4 0 1 0
"""

class GeometryInspectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.workspace = GhostWorkspace()
        self.tab = self.workspace.geometry_tab
        self.workspace.resize(1500, 900)
        self.workspace.show()
        self.addCleanup(self._dispose)
        self._load(THIN)

    def _dispose(self):
        self.workspace.solver_tab._is_solving = False
        self.tab._set_dirty(False)
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def _load(self, source):
        path = Path(self.folder.name) / "inspection.geo"
        path.write_text(source, encoding="ascii")
        with mock.patch("ghost_backend.ui.geometry.QFileDialog.getOpenFileName", return_value=(str(path), "")), mock.patch("ghost_backend.ui.geometry.QMessageBox.information"):
            self.assertTrue(self.tab.load_geo())
        self.tab.chk_show_normals.setChecked(False)
        self._draw()

    def _draw(self):
        self.app.processEvents()
        self.tab.canvas.draw()
        self.app.processEvents()
        self.tab.canvas.draw()

    def _view(self, xlim, ylim):
        self.tab.canvas.ax.set_xlim(*xlim)
        self.tab.canvas.ax.set_ylim(*ylim)
        self._draw()

    def _event(self, point, *, dx_px=0, ax=None):
        ax = self.tab.canvas.ax if ax is None else ax
        x, y = ax.transData.transform(point)
        x += dx_px
        xdata, ydata = ax.transData.inverted().transform((x, y))
        return SimpleNamespace(x=x, y=y, xdata=xdata, ydata=ydata, inaxes=ax, button=1, key=None)

    def _select(self, rows):
        self.tab.table.clearSelection()
        for row in rows:
            self.tab.table.selectionModel().select(self.tab.table.model().index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def _isolate(self, material):
        combo = self.tab.cmb_material_isolation
        index = combo.findData(material)
        self.assertGreaterEqual(index, 0)
        combo.setCurrentIndex(index)
        self._draw()

    def _measure_thin(self):
        self._view((0.8, 1.2), (-0.03, 0.04))
        self.tab.btn_measure_gap.setChecked(True)
        self.assertTrue(self.tab._inspection_button_press(self._event((1.0, 0.0))))
        self.assertIsNotNone(self.tab._gap_anchor)
        self.assertTrue(self.tab._inspection_button_press(self._event((1.1, 0.01))))
        self.assertIsNotNone(self.tab._gap_result)
        return self.tab._gap_result

    def _pixel(self, point):
        rgba = np.asarray(self.tab.canvas.buffer_rgba())
        x, y = self.tab.canvas.ax.transData.transform(point)
        return rgba[rgba.shape[0] - 1 - int(round(y)), int(round(x)), :3].astype(float)

    def _inset_magnification(self):
        main = self.tab.canvas.ax
        detail = self.tab._detail_ax
        p = np.array([[0.0, 0.0], [1.0, 0.0]])
        return np.linalg.norm(np.diff(detail.transData.transform(p), axis=0)) / np.linalg.norm(np.diff(main.transData.transform(p), axis=0))

    def test_two_click_local_gap_projects_onto_second_boundary(self):
        result = self._measure_thin()
        self.assertEqual(result["kind"], "Local gap")
        self.assertAlmostEqual(result["distance"], 0.01, places=12)
        np.testing.assert_allclose(result["first_point"], (1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(result["second_point"], (1.0, 0.01), atol=1e-12)
        self.assertEqual((result["first_row"], result["second_row"]), (0, 1))
        self.assertFalse(self.tab.is_dirty())
        self.tab.btn_clear_gap.click()
        self.assertIsNone(self.tab._gap_anchor)
        self.assertIsNone(self.tab._gap_result)

    def test_nearest_primitive_projection_clamps_to_endpoint(self):
        self._view((1.8, 2.1), (-0.03, 0.04))
        nearest = self.tab._nearest_primitive(self._event((2.0, 0.01), dx_px=2.0))
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest["row"], 1)
        np.testing.assert_allclose(nearest["point"], (2.0, 0.01), atol=1e-12)

    def test_minimum_selected_gap_reports_crossing_as_zero(self):
        self._load(CROSSED)
        self._select((0, 1))
        self.tab.btn_min_gap.click()
        result = self.tab._gap_result
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "Minimum gap")
        self.assertEqual(result["distance"], 0.0)
        np.testing.assert_allclose(result["first_point"], (1.0, 1.0), atol=1e-12)
        np.testing.assert_allclose(result["second_point"], (1.0, 1.0), atol=1e-12)
        self.assertFalse(self.tab.is_dirty())

    def test_material_isolation_filters_picks_even_with_context_visible(self):
        self._load(NESTED)
        self.tab.chk_isolation_context.setChecked(True)
        self._isolate("d1")
        self.assertEqual(self.tab._hit_test(self._event((-2.0, 0.2))), 0)
        self.assertEqual(self.tab._hit_test(self._event((-1.0, 0.2))), 1)
        self.assertIsNone(self.tab._hit_test(self._event((4.0, 0.2))))
        self.assertIsNone(self.tab._nearest_primitive(self._event((4.0, 0.2))))
        self.tab.chk_isolation_context.setChecked(False)
        self.assertIsNone(self.tab._hit_test(self._event((4.0, 0.2))))
        self._isolate(None)
        self.assertEqual(self.tab._hit_test(self._event((4.0, 0.2))), 2)

    def test_isolated_coating_does_not_paint_pec_hole(self):
        self._load(NESTED)
        self.tab.chk_fill_materials.setChecked(True)
        self.tab.chk_isolation_context.setChecked(False)
        self._isolate("d1")
        self.tab.canvas.ax.grid(False)
        self._view((-2.3, 2.3), (-2.3, 2.3))
        background = np.array(self.tab.canvas.ax.get_facecolor()[:3]) * 255
        np.testing.assert_allclose(self._pixel((0.31, 0.23)), background, atol=3)
        self.assertGreater(np.max(np.abs(self._pixel((1.51, 0.23)) - background)), 10)

    def test_geometry_and_solver_units_sync_without_editing_coordinates(self):
        original = copy.deepcopy(self.tab.segments)
        self.tab.set_geometry_units("meters")
        self.assertEqual(self.workspace.solver_tab.cmb_units.currentText(), "meters")
        self.workspace.solver_tab.cmb_units.setCurrentText("inches")
        self.assertEqual(self.tab.geometry_units(), "inches")
        self.assertEqual(self.tab.segments, original)
        self.assertFalse(self.tab.is_dirty())
        self.workspace.solver_tab._is_solving = True
        self.workspace.solver_tab._apply_job_state()
        self.tab.set_geometry_units("meters")
        self.assertEqual(self.tab.geometry_units(), "inches")
        self.assertEqual(self.workspace.solver_tab.cmb_units.currentText(), "inches")

    def test_detail_zoom_matches_actual_screen_magnification_after_main_zoom(self):
        self.tab.chk_detail_inset.setChecked(True)
        self._draw()
        self.assertIsNotNone(self.tab._detail_ax)
        for zoom in (20, 50):
            self.tab.cmb_detail_zoom.setCurrentIndex(self.tab.cmb_detail_zoom.findData(zoom))
            self._draw()
            self.assertAlmostEqual(self._inset_magnification(), zoom, delta=0.1)
        self._view((0.8, 1.2), (-0.03, 0.04))
        self.assertAlmostEqual(self._inset_magnification(), 50, delta=0.1)
        self.assertIn(self.tab._detail_ax, self.tab.canvas.ax.child_axes)

    def test_detail_toggle_and_geometry_reload_leave_one_live_inset(self):
        self.tab.chk_detail_inset.setChecked(True)
        self._draw()
        old = self.tab._detail_ax
        self.tab.chk_detail_inset.setChecked(False)
        self._draw()
        self.assertNotIn(old, self.tab.canvas.ax.child_axes)
        self.assertTrue(self.tab._detail_ax is None or not self.tab._detail_ax.get_visible())
        self.tab.chk_detail_inset.setChecked(True)
        self._load(NESTED)
        self.assertIsNotNone(self.tab._detail_ax)
        self.assertIn(self.tab._detail_ax, self.tab.canvas.ax.child_axes)
        self.assertEqual(len(self.tab.canvas.ax.child_axes), 1)
        self.assertAlmostEqual(self._inset_magnification(), self.tab.cmb_detail_zoom.currentData(), delta=0.1)

    def test_geometry_edit_clears_old_gap_measurement(self):
        self._measure_thin()
        self.tab.segments[1].y = [0.02, 0.02]
        self.tab._set_dirty(True)
        self.assertIsNone(self.tab._gap_anchor)
        self.assertIsNone(self.tab._gap_result)

    def _real_click(self, point, ax=None):
        ax = self.tab.canvas.ax if ax is None else ax
        x, y = ax.transData.transform(point)
        event = MouseEvent("button_press_event", self.tab.canvas, x, y, button=1)
        self.assertIs(event.inaxes, ax)
        self.tab.canvas.callbacks.process("button_press_event", event)
        self.tab.canvas.callbacks.process("button_release_event",
            MouseEvent("button_release_event", self.tab.canvas, x, y, button=1))
        self._draw()

    def test_real_clicks_measure_inside_inset_without_overview_picks(self):
        self._view((0, 2), (-0.8, 0.8))
        self.tab.chk_detail_inset.setChecked(True)
        self.tab._detail_center = (1, .005)
        self.tab._detail_dirty = True
        self._draw()
        self.tab.btn_measure_gap.setChecked(True)
        self._real_click((1, 0), self.tab._detail_ax)
        self.assertEqual(self.tab._gap_anchor["row"], 0)
        self.assertIsNone(self.tab._gap_result)
        self._real_click((1.005, .01), self.tab._detail_ax)
        self.assertEqual(self.tab._gap_result["first_row"], 0)
        self.assertEqual(self.tab._gap_result["second_row"], 1)
        self.assertAlmostEqual(self.tab._gap_result["distance"], .01, places=12)
        self.assertIsNone(self.tab._gap_anchor)
        self.tab.btn_measure_gap.setChecked(False)
        center = self.tab._detail_center
        self._real_click((1, .01), self.tab._detail_ax)
        self.assertEqual(self.tab._selected_row, 1)
        self.assertEqual(center, self.tab._detail_center)

    def test_real_overview_clicks_do_not_double_process_pick_event(self):
        self._view((.8, 1.2), (-.03, .04))
        self.tab.btn_measure_gap.setChecked(True)
        self._real_click((1, 0))
        self.assertIsNotNone(self.tab._gap_anchor)
        self.assertIsNone(self.tab._gap_result)
        self._real_click((1.1, .01))
        self.assertAlmostEqual(self.tab._gap_result["distance"], .01, places=12)
        self.assertIsNone(self.tab._gap_anchor)

    def test_inset_scroll_and_toolbar_do_not_move_overview(self):
        self.tab.chk_detail_inset.setChecked(True)
        self._draw()
        main, detail = self.tab.canvas.ax, self.tab._detail_ax
        limits = (main.get_xlim(), main.get_ylim())
        x, y = detail.transAxes.transform((.5, .5))
        event = MouseEvent("scroll_event", self.tab.canvas, x, y, button="up", step=1)
        self.tab.canvas.callbacks.process("scroll_event", event)
        self._draw()
        self.assertEqual(self.tab.cmb_detail_zoom.currentData(), 50)
        self.assertEqual(limits, (main.get_xlim(), main.get_ylim()))
        for toggle, state in ((self.tab.toolbar.pan, "_pan_info"),
                              (self.tab.toolbar.zoom, "_zoom_info")):
            toggle()
            self._real_click(self.tab._detail_center, detail)
            self.assertIsNone(getattr(self.tab.toolbar, state))
            toggle()
        self.assertEqual(limits, (main.get_xlim(), main.get_ylim()))

    def test_compatible_isolation_preserves_ruler_but_hidden_boundary_clears_it(self):
        self._load(NESTED)
        self._select((0, 1))
        self.tab.btn_min_gap.click()
        result = copy.deepcopy(self.tab._gap_result)
        self._isolate("d1")
        self.assertEqual(self.tab._gap_result, result)
        self.tab.chk_isolation_context.setChecked(False)
        self.assertEqual(self.tab._gap_result, result)
        self._isolate("d2")
        self.assertIsNone(self.tab._gap_result)

    def test_hit_testing_outside_axes_returns_no_primitive(self):
        event = MouseEvent("button_press_event", self.tab.canvas, -5, -5, button=1)
        self.assertIsNone(self.tab._nearest_primitive(event))

    def _pump_until(self, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.app.processEvents()
            if time.monotonic() >= deadline:
                self.fail("Timed out waiting for the gap worker's Qt signals.")
            time.sleep(0.001)
        self.app.processEvents()

    @contextmanager
    def _held_minimum_search(self):
        # 160 * 160 primitive pairs selects the real background-worker route.
        for segment in self.tab.segments:
            segment.x = segment.x * 160
            segment.y = segment.y * 160
        self._select((0, 1))
        entered, release = threading.Event(), threading.Event()
        captured = {}
        answer = dict(first_point=(0.0, 0.0), second_point=(0.0, 0.01),
                      distance=0.01, first_primitive=0, second_primitive=0)

        def blocked_compute(first, second, *, checkpoint):
            captured["segments"] = (first, second)
            captured["thread"] = threading.get_ident()
            entered.set()
            if not release.wait(10.0):
                raise RuntimeError("Test did not release the gap-worker barrier.")
            checkpoint()
            return copy.deepcopy(answer)

        with mock.patch("ghost_backend.ui.geometry_inspection.closest_segment_points",
                        side_effect=blocked_compute) as compute:
            self.tab.btn_measure_gap.setChecked(True)
            self.tab.btn_min_gap.click()
            worker = self.tab._gap_worker
            self.assertIsNotNone(worker)
            try:
                self._pump_until(entered.is_set)
                yield SimpleNamespace(worker=worker, version=worker.version,
                                      release=release, captured=captured,
                                      answer=answer, compute=compute)
            finally:
                worker.abort.set()
                release.set()
                self._pump_until(lambda: self.tab._gap_worker is None)

    def test_large_minimum_search_finishes_in_worker_with_copied_geometry(self):
        with self._held_minimum_search() as search:
            self.assertIsInstance(search.worker, _GapWorker)
            self.assertNotEqual(search.captured["thread"], threading.get_ident())
            self.assertFalse(self.tab.btn_measure_gap.isChecked())
            self.assertEqual(self.tab.btn_min_gap.text(), "Cancel min gap")
            self.assertIsNone(self.tab._gap_result)
            self.assertIsNot(search.captured["segments"][0], self.tab.segments[0])
            original = self.tab.segments[0].x[0]
            self.tab.segments[0].x[0] = 99.0
            self.assertEqual(search.captured["segments"][0].x[0], original)
            self.tab.segments[0].x[0] = original
            self.tab.chk_detail_inset.setChecked(True)
            self._draw()
            self.assertIsNotNone(self.tab._detail_ax)
            search.release.set()
            self._pump_until(lambda: self.tab._gap_worker is None)
            self.assertEqual(self.tab._gap_result["kind"], "Minimum gap")
            self.assertAlmostEqual(self.tab._gap_result["distance"], 0.01)
            self.assertEqual(self.tab.btn_min_gap.text(), "Min selected gap")
            self.assertTrue(self.tab.btn_min_gap.isEnabled())
            self.assertNotIn(search.worker, _GapWorker.active)
            search.compute.assert_called_once()
            self.assertFalse(self.tab.is_dirty())

    def test_large_minimum_search_cancel_discards_worker_result(self):
        with self._held_minimum_search() as search:
            self.tab.btn_min_gap.click()
            self.assertTrue(search.worker.abort.is_set())
            self.assertGreater(self.tab._gap_version, search.version)
            self.assertIn("cancel", self.tab.lbl_gap.text().lower())
            search.release.set()
            self._pump_until(lambda: self.tab._gap_worker is None)
            self.assertIsNone(self.tab._gap_result)
            self.assertIsNone(self.tab._gap_anchor)
            self.assertEqual(self.tab.btn_min_gap.text(), "Min selected gap")
            self.assertNotIn(search.worker, _GapWorker.active)
            self.assertFalse(self.tab.is_dirty())

    def test_local_measure_cancels_pending_search_and_rejects_late_signals(self):
        with self._held_minimum_search() as search:
            self.tab.btn_measure_gap.setChecked(True)
            self.assertTrue(search.worker.abort.is_set())
            self.assertGreater(self.tab._gap_version, search.version)
            local = copy.deepcopy(self._measure_thin())
            label = self.tab.lbl_gap.text()
            search.worker.ready.emit(search.version, ([0, 1], search.answer))
            search.worker.failed.emit(search.version, "Late minimum-gap failure")
            self.app.processEvents()
            self.assertEqual(self.tab._gap_result, local)
            self.assertEqual(self.tab.lbl_gap.text(), label)
            search.release.set()
            self._pump_until(lambda: self.tab._gap_worker is None)
            self.assertTrue(self.tab.btn_measure_gap.isChecked())
            self.assertEqual(self.tab._gap_result, local)
            self.assertEqual(self.tab.lbl_gap.text(), label)
            self.assertEqual(self.tab._gap_result["kind"], "Local gap")
            self.assertNotIn(search.worker, _GapWorker.active)

if __name__ == "__main__":
    unittest.main()
