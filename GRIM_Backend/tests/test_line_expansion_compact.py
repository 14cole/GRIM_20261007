"""Line Expansion stays usable inside the real, compact GRIM host."""

import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, QSettings, QSize, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLineEdit

from GRIM_Backend.ui.app import GrimCutWindow


class LineExpansionCompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_sys_path = sys.path[:]
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                cls.app.setFont(QFont(families[0], 9))

    @classmethod
    def tearDownClass(cls):
        cls.app.setFont(cls.original_font)
        sys.path[:] = cls.original_sys_path

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.temp.name) / "window.ini"), QSettings.IniFormat)
        self.window = GrimCutWindow(settings=settings)
        self.window.main_tabs.setCurrentWidget(self.window.ghost_integration)
        self.ghost = self.window.ghost_integration.workspace
        self.tab = self.ghost.line_expansion_tab
        self.ghost.setCurrentWidget(self.tab)
        self.window.resize(1200, 680)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.tab.job_is_running():
            self.tab.request_cancel()
            deadline = time.monotonic() + 5
            while self.tab.job_is_running() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.001)
        self.window.hide()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def assert_footer_visible(self):
        for control in (self.tab.run_button, self.tab.cancel_button, self.tab.progress):
            self.assertFalse(self.tab.controls_scroll.isAncestorOf(control))
            bounds = QRect(control.mapTo(self.tab, QPoint()), control.size())
            self.assertTrue(control.isVisible())
            self.assertTrue(self.tab.rect().contains(bounds), (bounds, self.tab.rect()))

    def test_compact_editor_scrolls_while_run_cancel_and_progress_remain_visible(self):
        self.tab._set_busy(True)
        for size in (QSize(1200, 680), QSize(1280, 720), QSize(1366, 768)):
            with self.subTest(size=size):
                self.window.resize(size)
                self.app.processEvents()
                self.assertEqual(self.window.size(), size)
                scroll = self.tab.controls_scroll
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                bar = scroll.verticalScrollBar()
                self.assertGreater(bar.maximum(), 0)
                for value in (bar.minimum(), bar.maximum()):
                    bar.setValue(value)
                    self.app.processEvents()
                    self.assert_footer_visible()
                scroll.ensureWidgetVisible(self.tab.output_edit)
                self.app.processEvents()
                bounds = QRect(self.tab.output_edit.mapTo(scroll.viewport(), QPoint()), self.tab.output_edit.size())
                self.assertTrue(scroll.viewport().rect().contains(bounds))

    def test_multiline_results_remain_readable_without_expanding_the_host(self):
        # Avoid the host trying to load our synthetic export notification.
        self.tab.files_exported.disconnect()
        self.tab.files_exported.connect(lambda *_: None)
        exported = QSignalSpy(self.tab.files_exported)
        stations = [f"Section {index}: station details" for index in range(100)]
        self.tab._on_finished({"output": "result.grim", "stations": stations, "warnings": ["Review sampling."]})
        self.app.processEvents()
        self.assertEqual(exported.count(), 1)
        self.assertEqual(exported.at(0), [["result.grim"], "line expansion"])
        self.assertEqual(self.window.size(), QSize(1200, 680))
        self.assertIn(stations[-1], self.tab.status_label.text())
        self.assertIn("Review sampling.", self.tab.status_label.text())
        self.assertLessEqual(self.tab.status_scroll.height(), 96)
        bar = self.tab.status_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum())
        self.app.processEvents()
        self.assertLessEqual(self.tab.status_label.mapTo(self.tab.status_scroll.viewport(), self.tab.status_label.rect().bottomLeft()).y(), self.tab.status_scroll.viewport().height())

    def test_section_and_corner_editing_survives_scrolling(self):
        module = sys.modules[type(self.tab).__module__]
        with patch.object(module.QFileDialog, "getOpenFileNames", return_value=(["section.geo"], "")):
            self.tab.add_button.click()
        self.assertEqual(self.tab.section_table.rowCount(), 1)
        table = self.tab.section_table
        # Exercise the actual cell editor at the horizontally scrolled end.
        self.tab.controls_scroll.ensureWidgetVisible(table)
        item = table.item(0, 12)
        table.scrollToItem(item)
        table.setCurrentItem(item)
        table.editItem(item)
        self.app.processEvents()
        editor = table.findChild(QLineEdit)
        self.assertIsNotNone(editor)
        editor.setText("1")
        QTest.keyClick(editor, Qt.Key_Return)
        self.app.processEvents()
        self.assertEqual(item.text(), "1")
        table.item(0, 10).setText("0")
        table.item(0, 11).setText("0")
        self.assertEqual(self.tab.sections()[0]["normal_end"], (0.0, 0.0, 1.0))
        self.tab.controls_scroll.ensureWidgetVisible(self.tab.add_corner_button)
        self.tab.add_corner_button.click()
        self.assertEqual(len(self.tab.corners()), 1)
        self.tab.corner_table.selectRow(0)
        self.tab.remove_corner_button.click()
        self.assertEqual(self.tab.corner_table.rowCount(), 0)
        table.selectRow(0)
        self.tab.remove_button.click()
        self.assertEqual(table.rowCount(), 0)

    def test_cancellation_from_fixed_footer_reaches_the_worker(self):
        module = sys.modules[type(self.tab).__module__]
        started = threading.Event()

        def slow_expansion(sections, *, cancel_check, progress_callback, **arguments):
            started.set()
            progress_callback(1, 2, "Working")
            deadline = time.monotonic() + 5
            while not cancel_check():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Test cancellation did not arrive")
                time.sleep(0.001)
            raise InterruptedError("Cancelled test expansion")

        self.tab._append_row(["section.geo"] + ["0"] * 8 + ["1"] + [""] * 3)
        self.tab.output_edit.setText(str(Path(self.temp.name) / "cancelled.grim"))
        with patch.object(module, "expand_wing_sections", side_effect=slow_expansion):
            self.tab.run_button.click()
            deadline = time.monotonic() + 5
            while not started.is_set() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.001)
            self.assertTrue(started.is_set())
            self.tab.controls_scroll.verticalScrollBar().setValue(0)
            self.app.processEvents()
            self.assert_footer_visible()
            self.assertFalse(self.tab.section_table.isEnabled())
            self.tab.cancel_button.click()
            deadline = time.monotonic() + 5
            while self.tab.job_is_running() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.001)
            self.assertFalse(self.tab.job_is_running())
        self.assertEqual(self.tab.status_label.text(), "Cancelled test expansion")
        self.assertTrue(self.tab.run_button.isEnabled())
        self.assertTrue(self.tab.section_table.isEnabled())
        self.assertFalse(Path(self.tab.output_edit.text()).exists())


if __name__ == "__main__":
    unittest.main()
