"""The dataset table hides its History column until the context menu shows it."""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QMenu

import GRIM_Backend.ui.app as grim_cut_gui
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.ui.dataset_actions import HISTORY_COLUMN, HISTORY_COLUMN_SETTINGS_KEY
from test_gui_shell import (
    _FakeFeatureWorkflow, _FakeFreddyIntegration, _FakeGhostIntegration,
    _MemorySettings, _RecordingWindow,
)

TOGGLE = "Show History Column"


class _ChoosingMenu(QMenu):
    """Stands in for QMenu: records the offered labels and picks one by text."""

    chosen = None
    labels: list[str] = []

    def exec(self, _position):
        type(self).labels = [action.text() for action in self.actions()]
        for action in self.actions():
            if action.text() == type(self).chosen:
                return action
        return None


class HistoryColumnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for name, replacement in (
            ("GhostIntegrationWidget", _FakeGhostIntegration),
            ("FreddyIntegrationWidget", _FakeFreddyIntegration),
        ):
            patch = mock.patch.object(grim_cut_gui, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(
            grim_cut_gui, "load_ghost_module", return_value=_FakeFeatureWorkflow()
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.settings = _MemorySettings()
        self.window = self._make_window(self.settings)

    def _make_window(self, settings):
        window = _RecordingWindow(settings=settings)
        grid = RcsGrid([0.0, 1.0], [0.0], [10.0], ["VV"], rcs=np.ones((2, 1, 1, 1), dtype=complex))
        window._add_dataset_row(grid, "Run 1", "loaded from test", file_name="")
        return window

    def _open_menu(self, window, position, choose=None):
        _ChoosingMenu.chosen = choose
        _ChoosingMenu.labels = []
        with mock.patch("GRIM_Backend.ui.dataset_actions.QMenu", _ChoosingMenu):
            window._on_dataset_context_menu(position)
        return list(_ChoosingMenu.labels)

    @staticmethod
    def _empty_area(table):
        return QPoint(5, table.rowViewportPosition(0) + table.rowHeight(0) + 5)

    @staticmethod
    def _row_position(table, row=0):
        return table.visualItemRect(table.item(row, 0)).center()

    def test_history_column_is_hidden_by_default_but_still_populated(self):
        table = self.window.table
        self.assertTrue(table.isColumnHidden(HISTORY_COLUMN))
        self.assertFalse(table.isColumnHidden(0))
        self.assertFalse(table.isColumnHidden(1))
        self.assertEqual(table.horizontalHeaderItem(HISTORY_COLUMN).text(), "History")
        self.assertEqual(table.item(0, HISTORY_COLUMN).text(), "loaded from test")

    def test_right_click_on_empty_area_offers_only_the_toggle(self):
        table = self.window.table
        table.clearSelection()
        labels = self._open_menu(self.window, self._empty_area(table))
        self.assertEqual(labels, [TOGGLE])
        self.assertEqual(table.selectionModel().selectedRows(), [])
        self.assertTrue(table.isColumnHidden(HISTORY_COLUMN))

    def test_toggle_shows_persists_and_hides_again(self):
        table = self.window.table
        table.clearSelection()
        self._open_menu(self.window, self._empty_area(table), choose=TOGGLE)
        self.assertFalse(table.isColumnHidden(HISTORY_COLUMN))
        self.assertIs(self.settings.value(HISTORY_COLUMN_SETTINGS_KEY), True)
        self.assertIn("shown", self.window.status.currentMessage())

        restored = self._make_window(self.settings)
        self.assertFalse(restored.table.isColumnHidden(HISTORY_COLUMN))

        table.selectRow(0)
        labels = self._open_menu(self.window, self._row_position(table), choose=TOGGLE)
        self.assertIn("Save", labels)
        self.assertIn("Delete", labels)
        self.assertIn(TOGGLE, labels)
        self.assertTrue(table.isColumnHidden(HISTORY_COLUMN))
        self.assertIs(self.settings.value(HISTORY_COLUMN_SETTINGS_KEY), False)

    def test_saved_string_setting_is_honoured(self):
        window = self._make_window(_MemorySettings({HISTORY_COLUMN_SETTINGS_KEY: "true"}))
        self.assertFalse(window.table.isColumnHidden(HISTORY_COLUMN))
        window = self._make_window(_MemorySettings({HISTORY_COLUMN_SETTINGS_KEY: "false"}))
        self.assertTrue(window.table.isColumnHidden(HISTORY_COLUMN))

    def test_right_click_on_a_row_still_selects_it_and_offers_row_actions(self):
        table = self.window.table
        table.clearSelection()
        labels = self._open_menu(self.window, self._row_position(table))
        self.assertEqual([index.row() for index in table.selectionModel().selectedRows()], [0])
        for label in ("Save", "Export as…", "Delete", "Text Color…", "Reset Text Color", TOGGLE):
            self.assertIn(label, labels)


if __name__ == "__main__":
    unittest.main()
