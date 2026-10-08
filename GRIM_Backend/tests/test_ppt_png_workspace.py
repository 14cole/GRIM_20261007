"""PNG export uses the complete reviewed report without PowerPoint setup."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_IMPORT_ERROR = None
try:
    import numpy as np
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    from GRIM_Backend.datasets.grid import RcsGrid
    from GRIM_Backend.reports import workspace as ppt_workspace
    from GRIM_Backend.reports.workspace import DatasetCatalogEntry, GUI_AVAILABLE, PptWorkspace
except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional GUI dependencies
    _IMPORT_ERROR = exc
    GUI_AVAILABLE = False


def _entry():
    shape = (3, 1, 7, 2)
    grid = RcsGrid(
        np.asarray((0.0, 90.0, 180.0)), np.asarray((0.0,)),
        np.arange(1.0, 8.0), np.asarray(("HH", "VV")),
        rcs_power=np.ones(shape), rcs_phase=np.zeros(shape),
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
               "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
               "angular_coordinate_system": "conic"},
    )
    return DatasetCatalogEntry("vehicle", "Vehicle", grid, "vehicle.grim")


@unittest.skipUnless(GUI_AVAILABLE, f"Report GUI dependencies unavailable: {_IMPORT_ERROR}")
class PptPngWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widgets = []

    def tearDown(self):
        for widget in reversed(self.widgets):
            self.wait_for_export(widget)
            widget.close()
            widget.deleteLater()
        QCoreApplication.processEvents()

    def workspace(self, **kwargs):
        widget = PptWorkspace(**kwargs)
        self.widgets.append(widget)
        return widget

    def ready(self, widget):
        widget.set_dataset_catalog((_entry(),))
        widget.select_frequencies(tuple(range(1, 8)))
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview(), widget.last_error)
        self.assertEqual(widget.preview_plan.plot_count, 7)
        self.assertEqual(len(widget.preview_plan.slides), 2)
        return widget.preview_plan

    def wait_for_export(self, widget):
        deadline = time.monotonic() + 5.0
        while widget._thread is not None and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.002)
        QCoreApplication.processEvents()
        self.assertIsNone(widget._thread, "Report export worker did not finish")

    def test_png_action_requires_a_current_successful_preview(self):
        widget = self.workspace(image_exporter=mock.Mock())
        self.assertEqual(widget.export_images_button.text(), "Export PNG images…")
        self.assertFalse(widget.export_images_button.isEnabled())
        with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory") as choose:
            self.assertFalse(widget.export_images())
            choose.assert_not_called()
        self.ready(widget)
        self.assertTrue(widget.export_images_button.isEnabled())
        widget.deck_title_edit.setText("Revised vehicle report")
        self.assertFalse(widget.export_images_button.isEnabled())
        with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory") as choose:
            self.assertFalse(widget.export_images())
            choose.assert_not_called()
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        with mock.patch.object(widget, "_build_plan", side_effect=ValueError("invalid cut")):
            self.assertFalse(widget.build_preview())
        self.assertFalse(widget.export_images_button.isEnabled())

    def test_folder_cancel_preserves_preview_and_creates_no_job(self):
        exporter = mock.Mock()
        widget = self.workspace(image_exporter=exporter)
        frozen = self.ready(widget)
        with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory", return_value=""):
            self.assertFalse(widget.export_images())
        exporter.assert_not_called()
        self.assertFalse(widget.job_is_running())
        self.assertIs(widget.preview_plan, frozen)
        self.assertTrue(widget.preview_is_current)
        self.assertIn("canceled", widget.status_label.text())
        self.assertTrue(widget.export_images_button.isEnabled())

    def test_exports_entire_frozen_plan_without_ppt_validation_and_preserves_ppt_state(self):
        started, release = threading.Event(), threading.Event()
        calls, images, presentations = [], [], []

        def export_images(plan, destination):
            calls.append((plan, Path(destination)))
            started.set()
            if not release.wait(3.0):
                raise RuntimeError("test release timed out")
            Path(destination).mkdir(parents=True, exist_ok=True)
            return Path(destination)

        widget = self.workspace(image_exporter=export_images, exporter=mock.Mock())
        widget.template_edit.setText("missing-and-invalid-template.txt")
        widget.output_edit.setText("invalid-powerpoint-output.csv")
        widget.deck_title_edit.setText("Vehicle / comparison: 3 GHz")
        frozen = self.ready(widget)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            widget.next_slide()
        self.assertEqual(widget.current_slide_index, 1)
        widget._last_exported_presentation = "previous.pptx"
        widget.open_presentation_button.setEnabled(True)
        widget.images_exported.connect(images.append)
        widget.report_exported.connect(presentations.append)
        with tempfile.TemporaryDirectory() as directory:
            try:
                with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory", return_value=directory):
                    self.assertTrue(widget.export_images())
                self.assertTrue(started.wait(2.0))
                self.assertTrue(widget.job_is_running())
                self.assertIn("PNG", widget.busy_operation())
                self.assertFalse(widget.controls_content.isEnabled())
                self.assertFalse(widget.export_button.isEnabled())
                self.assertFalse(widget.export_images_button.isEnabled())
                self.assertFalse(widget.build_preview())
                self.assertFalse(widget.export_report())
                with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory") as choose:
                    self.assertFalse(widget.export_images())
                    choose.assert_not_called()
                # Programmatic catalog/settings updates during a worker cannot
                # rewrite its reviewed immutable plan.
                widget.deck_title_edit.setText("Changed after export began")
            finally:
                release.set()
                self.wait_for_export(widget)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], frozen)
            self.assertEqual(calls[0][0].plot_count, 7)
            self.assertEqual(calls[0][1].parent, Path(directory))
            self.assertNotIn("/", calls[0][1].name)
            self.assertNotIn(":", calls[0][1].name)
            self.assertIn("_images_", calls[0][1].name)
            self.assertEqual(images, [str(calls[0][1])])
            self.assertEqual(presentations, [])
            self.assertIn("PNG", widget.status_label.text())
            self.assertEqual(widget._last_exported_presentation, "previous.pptx")
            self.assertTrue(widget.open_presentation_button.isEnabled())
            self.assertEqual(widget.output_edit.text(), "invalid-powerpoint-output.csv")
            self.assertFalse(widget.export_images_button.isEnabled())
            self.assertTrue(widget.controls_content.isEnabled())

    def test_repeated_exports_choose_new_child_folders_and_keep_previous_images(self):
        calls = []

        def exporter(plan, destination):
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path / "plot.png").write_bytes(b"previous export")
            calls.append(path)
            return path

        widget = self.workspace(image_exporter=exporter)
        self.ready(widget)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory", return_value=directory), \
                    mock.patch.object(ppt_workspace, "datetime") as clock:
                clock.now.return_value = datetime(2026, 10, 1, 12, 30, 0)
                for _ in range(2):
                    self.assertTrue(widget.export_images())
                    self.wait_for_export(widget)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertEqual(calls[1].name, calls[0].name + "_2")
            self.assertTrue(all(path.parent == Path(directory) for path in calls))
            self.assertEqual((calls[0] / "plot.png").read_bytes(), b"previous export")

    def test_settings_changed_during_folder_chooser_do_not_export_unreviewed_data(self):
        exporter = mock.Mock()
        widget = self.workspace(image_exporter=exporter)
        self.ready(widget)
        with tempfile.TemporaryDirectory() as directory:
            def choose(*args, **kwargs):
                widget.deck_title_edit.setText("Changed while choosing folder")
                return directory
            with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory", side_effect=choose):
                self.assertFalse(widget.export_images())
            exporter.assert_not_called()
            self.assertFalse(widget.job_is_running())
            self.assertFalse(widget.preview_is_current)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_worker_failure_restores_actions_and_does_not_signal_success(self):
        widget = self.workspace(image_exporter=mock.Mock(side_effect=OSError("PNG destination is not writable")))
        self.ready(widget)
        images, presentations = [], []
        widget.images_exported.connect(images.append)
        widget.report_exported.connect(presentations.append)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ppt_workspace.QFileDialog, "getExistingDirectory", return_value=directory):
                self.assertTrue(widget.export_images())
            self.wait_for_export(widget)
        self.assertIn("not writable", widget.last_error)
        self.assertEqual(images, [])
        self.assertEqual(presentations, [])
        self.assertTrue(widget.export_images_button.isEnabled())
        self.assertTrue(widget.export_button.isEnabled())
        self.assertTrue(widget.controls_content.isEnabled())
        self.assertIsNone(widget.busy_operation())

    def test_default_gui_export_works_when_powerpoint_packages_are_unavailable(self):
        # A fresh interpreter proves import-time independence as well as use of
        # the real default image exporter, without disturbing the test process.
        script = r'''
import builtins
real_import = builtins.__import__
def no_powerpoint(name, *args, **kwargs):
    if name.split('.')[0] in {'pptx', 'pythoncom', 'win32com'}:
        raise ImportError('PowerPoint package intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = no_powerpoint
import tempfile
from pathlib import Path
from unittest import mock
from GRIM_Backend.tests.test_ppt_png_workspace import PptPngWorkspaceTests
from GRIM_Backend.reports import workspace
case = PptPngWorkspaceTests()
case.setUpClass()
case.setUp()
try:
    widget = case.workspace()
    widget.template_edit.setText('missing-template.invalid')
    widget.output_edit.setText('')
    plan = case.ready(widget)
    completed = []
    widget.images_exported.connect(completed.append)
    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(workspace.QFileDialog, 'getExistingDirectory', return_value=directory):
            assert widget.export_images(), widget.last_error
        case.wait_for_export(widget)
        assert not widget.last_error, widget.last_error
        assert len(completed) == 1
        pngs = list(Path(completed[0]).rglob('*.png'))
        expected = plan.plot_count + sum(bool(slide.master_legend) for slide in plan.slides)
        assert len(pngs) == expected, (len(pngs), expected)
        assert all(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for path in pngs)
        assert widget._last_exported_presentation is None
        assert not list(Path(directory).rglob('*.pptx'))
        print('PNG export succeeded with PowerPoint imports blocked')
finally:
    case.tearDown()
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=40, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PNG export succeeded", result.stdout)


if __name__ == "__main__":
    unittest.main()
