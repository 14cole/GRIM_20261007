"""Wheel navigation must not silently change geometry table dropdowns."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QTableWidget
except ImportError:
    from PySide2.QtCore import QPoint, QPointF, Qt  # type: ignore
    from PySide2.QtGui import QWheelEvent  # type: ignore
    from PySide2.QtTest import QTest  # type: ignore
    from PySide2.QtWidgets import QApplication, QTableWidget  # type: ignore

from ghost_backend.ui.table_editors import ScrollSafeComboBox


class ScrollSafeComboBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.table = QTableWidget(60, 1)
        self.table.resize(260, 180)
        self.combo = ScrollSafeComboBox()
        self.combo.addItems(["Material %d" % number for number in range(50)])
        self.combo.setCurrentIndex(1)
        self.combo.setMaxVisibleItems(5)
        self.table.setCellWidget(0, 0, self.combo)
        self.table.show()
        self.table.activateWindow()
        self.app.processEvents()
        self.changes = []
        self.combo.currentIndexChanged.connect(self.changes.append)

    def tearDown(self):
        self.combo.hidePopup()
        self.table.close()
        self.table.deleteLater()
        self.app.processEvents()

    def wheel(self, target, *, angle=-120, pixels=0):
        pos = target.rect().center()
        event = QWheelEvent(
            QPointF(pos),
            QPointF(target.mapToGlobal(pos)),
            QPoint(0, pixels),
            QPoint(0, angle),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(target, event)
        self.app.processEvents()

    def test_closed_dropdown_scrolls_table_without_changing_material(self):
        for focused in (False, True):
            with self.subTest(focused=focused):
                self.table.verticalScrollBar().setValue(0)
                if focused:
                    self.combo.setFocus()
                else:
                    self.table.setFocus()
                self.app.processEvents()
                self.assertEqual(self.combo.hasFocus(), focused)
                self.wheel(self.combo)
                self.assertEqual(self.combo.currentIndex(), 1)
                self.assertEqual(self.changes, [])
                self.assertGreater(self.table.verticalScrollBar().value(), 0)

    def test_scrolling_up_and_at_table_edge_never_edits_material(self):
        scrollbar = self.table.verticalScrollBar()
        scrollbar.setValue(10)
        self.wheel(self.combo, angle=120)
        self.assertLess(scrollbar.value(), 10)
        scrollbar.setValue(0)
        self.wheel(self.combo, angle=120)
        self.assertEqual(scrollbar.value(), 0)
        self.assertEqual(self.combo.currentIndex(), 1)
        self.assertEqual(self.changes, [])

    def test_keyboard_selection_still_works(self):
        self.combo.setFocus()
        QTest.keyClick(self.combo, Qt.Key_Down)
        self.assertEqual(self.combo.currentIndex(), 2)
        self.assertEqual(self.changes, [2])

    def test_open_popup_scrolls_and_click_selects_material(self):
        QTest.mouseClick(self.combo, Qt.LeftButton)
        self.app.processEvents()
        view = self.combo.view()
        self.assertTrue(view.isVisible())
        self.assertGreater(view.verticalScrollBar().maximum(), 0)
        self.wheel(view.viewport())
        self.assertGreater(view.verticalScrollBar().value(), 0)
        self.assertEqual(self.table.verticalScrollBar().value(), 0)
        target_index = view.indexAt(view.viewport().rect().center())
        self.assertTrue(target_index.isValid())
        self.assertNotEqual(target_index.row(), 1)
        # Qt briefly suppresses popup release events after the opening click.
        QTest.qWait(200)
        QTest.mouseMove(view.viewport(), view.visualRect(target_index).center())
        QTest.mouseClick(
            view.viewport(), Qt.LeftButton,
            pos=view.visualRect(target_index).center(),
        )
        self.app.processEvents()
        self.assertEqual(self.combo.currentIndex(), target_index.row())
        self.assertEqual(self.changes, [target_index.row()])
        self.assertFalse(view.isVisible())

    def test_geometry_dropdown_wheels_preserve_materials_and_clean_state(self):
        from ghost_backend.geometry.io import Segment
        from ghost_backend.ui.geometry import GeometryTab

        geometry = GeometryTab()
        try:
            geometry.segments = [
                Segment(
                    "Segment %d" % row, "5", ["5", "0", "2", "1", "2"],
                    [float(row), float(row + 1)], [0.0, 1.0],
                )
                for row in range(40)
            ]
            geometry.ibcs_entries = [
                [str(flag), "linear", "50", "0", "100", "0"]
                for flag in range(1, 41)
            ]
            geometry.dielectric_entries = [
                [str(flag), str(flag + 1), "0", "1", "0"]
                for flag in range(1, 4)
            ]
            geometry.table.setRowCount(len(geometry.segments))
            geometry._populate_small_table(
                geometry.table_ibc, geometry.ibcs_entries,
                geometry.lbl_ibc, "IBCS/Resistances",
            )
            geometry._populate_small_table(
                geometry.table_diel, geometry.dielectric_entries,
                geometry.lbl_diel, "Dielectrics",
            )
            geometry._refresh_segment_dropdowns()
            geometry.resize(1400, 700)
            geometry.show()
            geometry.activateWindow()
            self.app.processEvents()

            properties = [list(segment.properties) for segment in geometry.segments]
            ibc_rows = geometry._read_small_table(geometry.table_ibc)
            dielectric_rows = geometry._read_small_table(geometry.table_diel)
            dirty_changes = []
            geometry_changes = []
            geometry.dirty_changed.connect(dirty_changes.append)
            geometry.geometry_changed.connect(lambda: geometry_changes.append(True))
            self.assertFalse(geometry.is_dirty())
            for table, column in (
                (geometry.table, 1),   # Segment type.
                (geometry.table, 3),   # IBC / resistance assignment.
                (geometry.table, 4),   # Positive-side material.
                (geometry.table, 5),   # Negative-side material (TYPE 5).
                (geometry.table_ibc, 1),  # IBC taper kind.
            ):
                with self.subTest(table=table is geometry.table, column=column):
                    scrollbar = table.verticalScrollBar()
                    scrollbar.setValue(0)
                    combo = table.cellWidget(0, column)
                    self.assertIsInstance(combo, ScrollSafeComboBox)
                    self.assertTrue(combo.isEnabled())
                    index = combo.currentIndex()
                    combo.setFocus()
                    self.app.processEvents()
                    self.assertTrue(combo.hasFocus())
                    self.wheel(combo)
                    self.assertGreater(scrollbar.value(), 0)
                    self.assertEqual(combo.currentIndex(), index)
                    self.assertEqual(
                        [segment.properties for segment in geometry.segments],
                        properties,
                    )
                    self.assertEqual(
                        geometry._read_small_table(geometry.table_ibc), ibc_rows,
                    )
                    self.assertEqual(
                        geometry._read_small_table(geometry.table_diel),
                        dielectric_rows,
                    )
                    self.assertEqual(geometry.ibcs_entries, ibc_rows)
                    self.assertEqual(geometry.dielectric_entries, dielectric_rows)
                    self.assertFalse(geometry.is_dirty())
                    self.assertEqual(dirty_changes, [])
                    self.assertEqual(geometry_changes, [])
        finally:
            geometry.close()
            geometry.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
