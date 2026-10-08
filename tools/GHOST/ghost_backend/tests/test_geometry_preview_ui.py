"""Screen-space normals, close-edge selection, and finding navigation."""
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
from matplotlib.quiver import Quiver
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.geometry import GeometryTab

GEOMETRY = """Title: Thin boundaries
Segment: lower 2
properties: 2 10 0 0 0
0 0 2 0
Segment: upper 2
properties: 2 10 0 0 0
0 0.01 2 0.01
Segment: remote 2
properties: 2 10 0 0 0
5 1 6 1
IBCS_Resistances:
Dielectrics:
"""

class GeometryPreviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name) / "thin.geo"
        path.write_text(GEOMETRY, encoding="ascii")
        self.tab = GeometryTab()
        self.tab.resize(1400, 850)
        self.tab.show()
        self.addCleanup(self._dispose)
        with mock.patch("ghost_backend.ui.geometry.QFileDialog.getOpenFileName", return_value=(str(path), "")), mock.patch("ghost_backend.ui.geometry.QMessageBox.information"):
            self.assertTrue(self.tab.load_geo())
        self.tab.chk_show_normals.setChecked(True)
        self.tab.cmb_normal_scope.setCurrentIndex(self.tab.cmb_normal_scope.findData("all"))
        self.app.processEvents()
        self.tab.canvas.draw()

    def _dispose(self):
        self.tab._set_dirty(False)
        self.tab.close()
        self.tab.deleteLater()
        self.app.processEvents()

    def _quiver(self):
        return next(artist for artist in self.tab.normal_artists if isinstance(artist, Quiver))

    def _pixel_lengths(self):
        q = self._quiver()
        origins = np.column_stack((q.X, q.Y))
        tips = origins + np.column_stack((q.U, q.V))
        transform = self.tab.canvas.ax.transData
        return np.linalg.norm(transform.transform(tips) - transform.transform(origins), axis=1)

    def test_normal_size_stays_readable_after_zoom_and_draw(self):
        np.testing.assert_allclose(self._pixel_lengths(), 18.0, atol=1e-8)
        before = self._quiver()
        self.tab.canvas.ax.set_xlim(-0.1, 2.1)
        self.tab.canvas.ax.set_ylim(-0.2, 0.2)
        self.tab.canvas.draw()
        self.app.processEvents()
        self.assertIsNot(before, self._quiver())
        np.testing.assert_allclose(self._pixel_lengths(), 18.0, atol=1e-8)
        self.assertEqual(self.tab._normal_view, self.tab._preview_view_key())

    def test_selected_only_tracks_multiselection_and_clearing(self):
        self.tab.cmb_normal_scope.setCurrentIndex(self.tab.cmb_normal_scope.findData("selected"))
        self.tab.table.clearSelection()
        self.assertFalse(any(isinstance(a, Quiver) for a in self.tab.normal_artists))
        for row in (0, 2):
            self.tab.table.selectionModel().select(self.tab.table.model().index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        q = self._quiver()
        self.assertEqual(set(zip(q.X, q.Y)), {(1.0, 0.0), (5.5, 1.0)})
        self.tab.table.clearSelection()
        self.assertIsNone(self.tab._selected_row)
        self.assertFalse(any(isinstance(a, Quiver) for a in self.tab.normal_artists))

    def test_nearest_boundary_wins_when_pick_artist_is_another_close_edge(self):
        self.tab.canvas.ax.set_xlim(-0.1, 2.1)
        self.tab.canvas.ax.set_ylim(-0.1, 0.1)
        self.tab.canvas.draw()
        x, y = self.tab.canvas.ax.transData.transform((1.0, 0.009))
        mouse = SimpleNamespace(x=x, y=y, inaxes=self.tab.canvas.ax)
        self.assertEqual(self.tab._hit_test(mouse), 1)
        self.tab._on_plot_pick(SimpleNamespace(artist=self.tab.segment_lines[0], mouseevent=mouse))
        self.assertEqual({i.row() for i in self.tab.table.selectedIndexes()}, {1})

    def test_zoomed_long_primitive_gets_normal_inside_visible_slice(self):
        self.tab.canvas.ax.set_xlim(0.1, 0.3)
        self.tab.canvas.ax.set_ylim(-0.02, 0.03)
        self.tab.canvas.draw()
        self.app.processEvents()
        q = self._quiver()
        self.assertEqual(len(q.X), 2)
        self.assertTrue(np.all((q.X >= 0.1) & (q.X <= 0.3)))
        np.testing.assert_allclose(self._pixel_lengths(), 18.0, atol=1e-8)

    def test_finding_click_uses_source_row_after_severity_sort(self):
        findings = [("WARN", 0, "Check lower edge"), ("ERROR", 2, "Remote material mismatch")]
        with mock.patch("ghost_backend.ui.geometry.QMessageBox.warning"):
            self.tab._validation_ready(self.tab._validation_version, (findings, {0, 2}))
        self.assertEqual(self.tab.validation_results.item(0, 2).text(), "Remote material mismatch")
        self.tab._select_validation_finding(0, 2)
        self.assertEqual({i.row() for i in self.tab.table.selectedIndexes()}, {2})
        lo, hi = self.tab.canvas.ax.get_xlim()
        self.assertLess(lo, 5)
        self.assertGreater(hi, 6)
        self.assertGreater(lo, 4)

    def test_bor_mixed_sheet_preview_uses_solver_material_sides(self):
        self.tab.segments[0].properties[0] = "1"
        self.tab.set_geometry_mode("bor")
        front, _, back, _ = self.tab._segment_side_materials(self.tab.segments[1])
        self.assertEqual((front, back), ("air", "air"))
        self.tab.set_geometry_mode("2d")
        front, _, back, _ = self.tab._segment_side_materials(self.tab.segments[1])
        self.assertEqual((front, back), ("air", "PEC"))

    def test_mode_change_invalidates_old_findings(self):
        version = self.tab._validation_version
        with mock.patch("ghost_backend.ui.geometry.QMessageBox.warning"):
            self.tab._validation_ready(version, ([("WARN", 0, "Old planar finding")], {0}))
        self.tab.set_geometry_mode("bor")
        self.assertEqual(self.tab.validation_results.rowCount(), 0)
        self.assertEqual(self.tab.issue_rows, set())
        self.tab._validation_ready(version, ([("ERROR", 2, "Stale result")], {2}))
        self.assertEqual(self.tab.validation_results.rowCount(), 0)
        self.assertEqual(self.tab.issue_rows, set())

if __name__ == "__main__":
    unittest.main()
