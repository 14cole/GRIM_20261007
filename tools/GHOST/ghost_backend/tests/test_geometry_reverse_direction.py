"""Direction edits preserve segment shape and update the live normal preview."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib.quiver import Quiver
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from ghost_backend.geometry.io import parse_geometry, snapshot_to_geometry_text
from ghost_backend.ui.geometry import GeometryTab


GEOMETRY = """Title: Direction test
Segment: coating 2
properties: 2 -20 7 0 0
0 0 2 0
2 0 2 1
Segment: shared_coating 2
properties: 2 5 7 0 0
3 0 3 1
Segment: interface 5
properties: 5 0 0 1 2
4 0 5 0
5 0 5 1
5 1 4 0
IBCS_Resistances:
7 linear 10 2 90 4
Dielectrics:
1 3 0 1 0
2 4 0 1 0
"""


class ReverseSegmentDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "direction.geo"
        self.path.write_text(GEOMETRY, encoding="ascii")
        self.tab = GeometryTab()
        self.addCleanup(self._dispose_tab)
        with mock.patch("ghost_backend.ui.geometry.QFileDialog.getOpenFileName",
                        return_value=(str(self.path), "")), \
                mock.patch("ghost_backend.ui.geometry.QMessageBox.information"):
            self.assertTrue(self.tab.load_geo())

    def _dispose_tab(self):
        self.tab.deleteLater()
        self.app.processEvents()

    def _select(self, rows):
        self.tab.table.clearSelection()
        for row in rows:
            self.tab.table.selectionModel().select(
                self.tab.table.model().index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )

    def _normal_vectors(self):
        quiver = next(artist for artist in self.tab.normal_artists
                      if isinstance(artist, Quiver))
        return {
            (float(x), float(y)): np.array([u, v])
            for x, y, u, v in zip(quiver.X, quiver.Y, quiver.U, quiver.V)
        }

    def test_selected_open_and_closed_chains_reverse_and_round_trip(self):
        original = copy.deepcopy(self.tab.segments)
        original_materials = copy.deepcopy(self.tab._read_small_table(self.tab.table_ibc))
        self._select([0, 2])
        changed = []
        self.tab.geometry_changed.connect(lambda: changed.append(True))
        self.tab.issue_rows.update([0, 2])

        self.tab.btn_reverse_selected.click()

        self.assertTrue(self.tab.is_dirty())
        self.assertEqual(changed, [True])
        self.assertEqual(self.tab.issue_rows, set())
        self.assertEqual({item.row() for item in self.tab.table.selectedIndexes()}, {0, 2})
        for row in (0, 2):
            self.assertEqual(self.tab.segments[row].x, original[row].x[::-1])
            self.assertEqual(self.tab.segments[row].y, original[row].y[::-1])
            self.assertEqual(self.tab.segments[row].properties, original[row].properties)
        self.assertEqual(self.tab.segments[1], original[1])
        self.assertEqual(self.tab._read_small_table(self.tab.table_ibc), original_materials)
        self.assertIn("tapers", self.tab.lbl_status.text())
        self.assertEqual(list(self.tab.segment_lines[0].get_xdata()), [2.0, 2.0, 0.0])
        self.assertEqual(list(self.tab.segment_lines[0].get_ydata()), [1.0, 0.0, 0.0])

        saved = snapshot_to_geometry_text(self.tab.get_geometry_snapshot())
        _title, reloaded, ibcs, _dielectrics = parse_geometry(saved)
        self.assertEqual(reloaded, self.tab.segments)
        self.assertEqual(ibcs, original_materials)

        self.tab.btn_reverse_selected.click()
        self.assertEqual(self.tab.segments, original)
        self.assertEqual(changed, [True, True])

    def test_visible_normals_flip_at_same_locations_and_overlays_refresh(self):
        self.tab.chk_show_normals.setChecked(True)
        self.tab.chk_show_impedance.setChecked(True)
        self.tab.chk_fill_materials.setChecked(True)
        before = self._normal_vectors()
        self._select([0, 2])
        with mock.patch.object(self.tab, "_render_impedance_overlay",
                               wraps=self.tab._render_impedance_overlay) as impedance, \
                mock.patch.object(self.tab, "_render_fills", wraps=self.tab._render_fills) as fills:
            self.tab.btn_reverse_selected.click()
        after = self._normal_vectors()

        self.assertEqual(before.keys(), after.keys())
        for midpoint in before:
            expected = before[midpoint] if midpoint == (3.0, 0.5) else -before[midpoint]
            np.testing.assert_allclose(after[midpoint], expected)
        impedance.assert_called_once_with()
        fills.assert_called_once_with()

    def test_no_selection_does_not_change_geometry(self):
        original = copy.deepcopy(self.tab.segments)
        self.tab.table.clearSelection()
        self.tab.btn_reverse_selected.click()
        self.assertEqual(self.tab.segments, original)
        self.assertFalse(self.tab.is_dirty())
        self.assertIn("Select segments", self.tab.lbl_status.text())

    def test_incomplete_pairs_leave_every_selected_segment_unchanged(self):
        self.tab.segments[2].x.append(7.0)
        original = copy.deepcopy(self.tab.segments)
        self._select([0, 2])
        with mock.patch("ghost_backend.ui.geometry.QMessageBox.warning") as warning:
            self.tab.btn_reverse_selected.click()
        warning.assert_called_once()
        self.assertEqual(self.tab.segments, original)
        self.assertFalse(self.tab.is_dirty())


if __name__ == "__main__":
    unittest.main()
