"""Focused GUI regressions for the independent Plotting/ISAR settings popups."""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QScrollArea, QWidget

import GRIM_Backend.ui.app as grim_cut_gui
from GRIM_Backend.plotting.models import PlotContext
from GRIM_Backend.plotting.session import plot_context_facade


@plot_context_facade(PlotContext)
class _SettingsBuilderWindow(QMainWindow):
    """Construct only the QWidget base needed by the settings/context builder."""

    _build_plot_left_context = grim_cut_gui.GrimCutWindow._build_plot_left_context
    _activate_plot_tab = grim_cut_gui.GrimCutWindow._activate_plot_tab
    _clear_zoom_box_drag = grim_cut_gui.GrimCutWindow._clear_zoom_box_drag
    _clear_pan_drag = grim_cut_gui.GrimCutWindow._clear_pan_drag

    def __init__(self) -> None:
        super().__init__()
        self._plot_controls_by_tab: dict[str, dict] = {}

    def _schedule_hover(self, *_args) -> None:
        pass

    def _reset_hover_readout(self, *_args) -> None:
        pass

    def _move_shared_right_panel(self, _tab_key: str) -> None:
        pass


class PlotSettingsUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = _SettingsBuilderWindow()
        self.plot_panel = QWidget(self.window)
        self.isar_panel = QWidget(self.window)
        self.plot_context = self.window._build_plot_left_context(
            self.plot_panel, "plotting"
        )
        self.isar_context = self.window._build_plot_left_context(
            self.isar_panel, "isar"
        )

    def tearDown(self) -> None:
        self.plot_context.settings_frame.close()
        self.isar_context.settings_frame.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _label(popup, text: str) -> QLabel:
        return next(
            label
            for label in popup.content_widget.findChildren(QLabel)
            if label.text() == text
        )

    def test_popups_have_independent_titles_scroll_views_and_isar_content(self) -> None:
        plot_popup = self.plot_context.settings_frame
        isar_popup = self.isar_context.settings_frame

        self.assertEqual(plot_popup.windowTitle(), "Plot Settings")
        self.assertEqual(isar_popup.windowTitle(), "ISAR Settings")
        self.assertEqual(self.plot_context.btn_settings.text(), "Plot Settings")
        self.assertEqual(self.isar_context.btn_settings.text(), "ISAR Settings")

        for popup in (plot_popup, isar_popup):
            self.assertIsInstance(popup.scroll_area, QScrollArea)
            self.assertIs(popup.scroll_area.widget(), popup.content_widget)
            self.assertTrue(popup.scroll_area.widgetResizable())
            self.assertEqual(
                popup.scroll_area.verticalScrollBarPolicy(), Qt.ScrollBarAsNeeded
            )
            self.assertGreaterEqual(popup.minimumWidth(), 420)
            self.assertGreaterEqual(popup.minimumHeight(), 280)

        plot_section = plot_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )
        isar_section = isar_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )
        self.assertIsNotNone(plot_section)
        self.assertIsNotNone(isar_section)
        self.assertTrue(plot_section.isHidden())
        self.assertFalse(isar_section.isHidden())

        plot_popup.show()
        isar_popup.resize(480, 300)
        isar_popup.show()
        self.app.processEvents()
        self.assertFalse(plot_section.isVisible())
        self.assertTrue(isar_section.isVisible())
        self.assertTrue(self.isar_context.combo_isar_recon.isVisible())
        self.assertFalse(
            hasattr(self.isar_context, "chk_isar_legacy_phase_attestation")
        )
        self.assertTrue(self._label(isar_popup, "Image dB").isVisible())
        self.assertFalse(self._label(isar_popup, "Plot X Min").isVisible())
        self.assertGreater(isar_popup.scroll_area.verticalScrollBar().maximum(), 0)

    def test_tab_switch_preserves_independent_settings_and_closes_previous_popup(self) -> None:
        plot_popup = self.plot_context.settings_frame
        isar_popup = self.isar_context.settings_frame
        plot_section = plot_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )

        plot_popup.show()
        self.app.processEvents()
        self.assertFalse(plot_section.isVisible())
        self.assertTrue(plot_section.isHidden())

        self.plot_context.spin_plot_xmin.setValue(-12.0)
        self.isar_context.spin_plot_xmin.setValue(-34.0)
        self.plot_context.btn_settings.setChecked(True)

        self.window._plot_contexts = {
            "plotting": self.plot_context,
            "isar": self.isar_context,
        }
        self.window._active_plot_tab = "plotting"
        self.window._dataset_ops_panel = QWidget(self.window)
        self.window._dataset_ops_visible = False
        with mock.patch.object(self.window, "_move_shared_right_panel"):
            self.window._activate_plot_tab("isar")

        self.app.processEvents()
        self.assertFalse(plot_popup.isVisible())
        self.assertFalse(self.plot_context.btn_settings.isChecked())
        self.assertIs(self.window.combo_isar_recon, self.isar_context.combo_isar_recon)
        self.assertEqual(self.plot_context.spin_plot_xmin.value(), -12.0)
        self.assertEqual(self.window.spin_plot_xmin.value(), -34.0)


if __name__ == "__main__":
    unittest.main()
