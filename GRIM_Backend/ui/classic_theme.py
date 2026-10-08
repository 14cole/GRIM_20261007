"""Optional "Windows Classic" chrome: square corners, 3-D bevels, navy selection.

This module is deliberately self-contained so the look can be removed cleanly.
To remove the Windows Classic theme entirely:

1. Delete this file and the adjacent ``classic_icons`` directory.
2. Delete the ``"Windows Classic"`` entry (and its description line) in
   ``GRIM_Backend/ui/palette.py``.
3. Delete the ``classic_qss_overrides`` import and the ``if`` block at the end
   of ``build_qss`` in ``GRIM_Backend/ui/theme.py``.
4. Delete the ``classic_qss_overrides`` import and the ``if`` block after
   ``apply_host_theme`` in ``GRIM_Backend/integrations/freddy.py``.
5. Drop ``"Windows Classic"`` from the palette-name list in
   ``GRIM_Backend/tests/test_gui_shell.py``.
6. Delete ``GRIM_Backend/tests/test_classic_theme.py``.

Nothing else in the application references this module. A palette opts in by
carrying ``"ui_style": "classic"``; every other palette is unaffected.

The rules below are appended *after* the standard stylesheet, so they win on
equal specificity and only need to restate what the classic look changes:
border radius, bevel borders, white input fields, and classic widgets
(segmented progress bar, beveled scrollbars, yellow tooltips).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

CLASSIC_UI_STYLE = "classic"


def _glyph_path(name: str) -> str:
    """QSS needs real image paths; bundled glyphs require no cache writes.

    Keep these assets beside the module when distributing the application.
    The fixed black/white glyphs match the Windows Classic palette.
    """
    path = Path(__file__).resolve().with_name("classic_icons") / (name + ".svg")
    return path.as_posix().replace('"', '\\"')


def is_classic_palette(palette: Mapping[str, object]) -> bool:
    """Return True when ``palette`` asks for the Windows Classic chrome."""

    return str(palette.get("ui_style", "")) == CLASSIC_UI_STYLE


def classic_qss_overrides(palette: Mapping[str, object]) -> str:
    """Return QSS that restyles the standard stylesheet as early-2000s Windows."""

    face = str(palette["win_bg"])
    field = str(palette.get("field_bg", "#ffffff"))
    text = str(palette["text"])
    light = str(palette.get("bevel_light", "#ffffff"))
    shadow = str(palette["border"])
    dark = str(palette.get("bevel_dark", "#404040"))
    navy = str(palette["checked_bg"])
    grid = str(palette["grid"])
    muted = str(palette["muted"])
    checked_face = str(palette.get("checked_face", "#e8e6e0"))
    raised = f"{light} {dark} {dark} {light}"
    sunken = f"{shadow} {light} {light} {shadow}"
    field_sunken = f"{shadow} {light} {light} {shadow}"
    check_mark = _glyph_path("check")
    selected_check = _glyph_path("check_selected")
    arrow_up = _glyph_path("arrow_up")
    arrow_down = _glyph_path("arrow_down")
    arrow_left = _glyph_path("arrow_left")
    arrow_right = _glyph_path("arrow_right")
    return f"""
    /* ---- Windows Classic overrides (see classic_theme.py to remove) ---- */
    /* Font is set per widget type (QSS fonts do not cascade). Item views and
       QHeaderView are deliberately excluded: restyling their font trips a
       super().headerData() recursion in PySide6 6.11 inside FREDDY models. */
    QMenuBar, QMenu, QPushButton, QToolButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QRadioButton, QGroupBox, QTabBar, QStatusBar, QToolTip {{
        font-family: "Tahoma", "MS Sans Serif", "Microsoft Sans Serif", sans-serif;
    }}
    QMainWindow, QDialog, QWidget#dockBody, QScrollArea#controlDock {{ background: {face}; }}
    QToolTip {{ background: #ffffe1; color: #000000; border: 1px solid #000000; padding: 2px 4px; }}
    QSplitter::handle {{ background: {face}; }}

    QMenuBar {{ background: {face}; border-bottom: 1px solid {light}; }}
    QMenuBar::item {{ padding: 3px 8px; }}
    QMenuBar::item:selected {{ background: {navy}; color: white; }}
    QMenu {{
        background: {face}; border: 2px solid; border-color: {raised}; padding: 2px;
    }}
    QMenu::item {{ padding: 4px 24px 4px 22px; }}
    QMenu::item:selected {{ background: {navy}; color: white; }}
    QMenu::separator {{ height: 2px; margin: 2px 1px; border-top: 1px solid {shadow}; border-bottom: 1px solid {light}; }}
    QMenu::indicator:checked {{ image: url("{check_mark}"); background: transparent; width: 10px; height: 10px; }}
    QMenu::indicator:checked:selected {{ image: url("{selected_check}"); }}
    QStatusBar {{ background: {face}; color: {text}; border-top: 1px solid {light}; }}
    QStatusBar::item {{ border: 1px solid; border-color: {sunken}; }}

    QFrame {{ background: {face}; border: 2px groove {shadow}; border-radius: 0px; }}
    QLabel {{ background: transparent; border: none; border-radius: 0px; padding: 1px 2px; }}
    QFrame#paramSeparator {{ border: none; min-width: 2px; max-width: 2px; border-left: 1px solid {shadow}; border-right: 1px solid {light}; background: transparent; }}
    QGroupBox {{ border: 2px groove {shadow}; border-radius: 0px; margin-top: 8px; padding-top: 4px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; background: {face}; }}

    QToolButton, QPushButton {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised};
        border-radius: 0px; padding: 3px 9px;
    }}
    QToolButton:hover, QPushButton:hover {{ border-color: {raised}; }}
    QToolButton:pressed, QPushButton:pressed {{ border-color: {sunken}; padding: 4px 8px 2px 10px; }}
    QToolButton:checked {{ background: {checked_face}; color: {text}; border-color: {sunken}; }}
    QToolButton:focus, QPushButton:focus {{ border: 2px solid; border-color: {raised}; outline: 1px dotted {text}; }}
    QToolButton:disabled, QPushButton:disabled {{ background: {face}; color: {muted}; border-color: {raised}; }}
    /* FREDDY's embedded navigation rail (its own stylesheet, re-applied by the
       GRIM integration hook). */
    QToolButton#ModeNavButton {{
        border: 2px solid; border-color: {raised}; border-radius: 0px; padding: 6px 10px;
    }}
    QToolButton#ModeNavButton:hover {{ background: {face}; border-color: {raised}; }}
    QToolButton#ModeNavButton:checked {{ background: {navy}; color: white; border-color: {sunken}; }}
    QPushButton#featureWorkflowAction[primaryAction="true"] {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised}; font-weight: bold;
    }}
    QPushButton#featureWorkflowAction[primaryAction="true"]:hover {{ border: 2px solid; border-color: {raised}; }}

    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit, QListWidget QLineEdit {{
        background: {field}; color: {text}; border: 2px solid; border-color: {field_sunken};
        border-radius: 0px; padding: 2px 3px;
        selection-background-color: {navy}; selection-color: white;
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
        border: 2px solid; border-color: {field_sunken};
    }}
    QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
        background: {face}; color: {muted}; border-color: {field_sunken};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding; subcontrol-position: top right; width: 16px;
        background: {face}; border: 2px solid; border-color: {raised};
    }}
    QComboBox::down-arrow {{ image: url("{arrow_down}"); width: 8px; height: 8px; }}
    QComboBox QAbstractItemView {{
        background: {field}; color: {text}; border: 1px solid {text};
        selection-background-color: {navy}; selection-color: white;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
        width: 14px; background: {face}; border: 2px solid; border-color: {raised};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{arrow_up}"); width: 7px; height: 7px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{arrow_down}"); width: 7px; height: 7px; }}

    QCheckBox::indicator {{
        width: 13px; height: 13px; border-radius: 0px; background: {field};
        border: 2px solid; border-color: {field_sunken};
    }}
    QCheckBox::indicator:checked {{ background: {field}; border-color: {field_sunken}; image: url("{check_mark}"); }}
    QCheckBox::indicator:disabled {{ background: {face}; }}

    QTabWidget::pane {{ background: {face}; border: 2px solid; border-color: {raised}; top: -1px; }}
    QTabBar::tab {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised};
        border-bottom: none; border-radius: 0px; padding: 4px 10px; margin-right: 0px;
    }}
    QTabBar::tab:selected {{ background: {face}; border-color: {raised}; margin-top: -2px; padding-bottom: 6px; }}
    QTabBar::tab:hover {{ background: {face}; }}

    QTableWidget, QTableView, QListWidget, QTreeWidget, QTreeWidget#featureReadinessChecklist {{
        background: {field}; alternate-background-color: {field}; color: {text};
        border: 2px solid; border-color: {field_sunken}; border-radius: 0px; gridline-color: {grid};
    }}
    QTreeWidget::item, QListWidget::item {{ border-bottom: none; padding: 2px 4px; }}
    QTreeWidget::item:selected, QListWidget::item:selected,
    QTableWidget::item:selected, QTableView::item:selected {{ background: {navy}; color: white; border-bottom: none; }}
    QTreeWidget::branch {{ background: {field}; }}
    QTreeWidget::branch:has-children:!open {{ image: url("{arrow_right}"); }}
    QTreeWidget::branch:has-children:open {{ image: url("{arrow_down}"); }}
    /* The assembly tree draws its own expand control. */
    QTreeWidget#assemblyTree::branch:has-children {{ image: none; }}
    QHeaderView::section {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised}; padding: 3px 6px;
    }}

    QScrollBar:vertical {{ background: {checked_face}; width: 16px; margin: 16px 0 16px 0; border: none; }}
    QScrollBar:horizontal {{ background: {checked_face}; height: 16px; margin: 0 16px 0 16px; border: none; }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {face}; border: 2px solid; border-color: {raised}; min-height: 20px; min-width: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        background: {face}; border: 2px solid; border-color: {raised}; height: 16px; subcontrol-origin: margin;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        background: {face}; border: 2px solid; border-color: {raised}; width: 16px; subcontrol-origin: margin;
    }}
    QScrollBar::sub-line:vertical {{ subcontrol-position: top; }}
    QScrollBar::add-line:vertical {{ subcontrol-position: bottom; }}
    QScrollBar::sub-line:horizontal {{ subcontrol-position: left; }}
    QScrollBar::add-line:horizontal {{ subcontrol-position: right; }}
    QScrollBar::up-arrow:vertical {{ image: url("{arrow_up}"); width: 8px; height: 8px; }}
    QScrollBar::down-arrow:vertical {{ image: url("{arrow_down}"); width: 8px; height: 8px; }}
    QScrollBar::left-arrow:horizontal {{ image: url("{arrow_left}"); width: 8px; height: 8px; }}
    QScrollBar::right-arrow:horizontal {{ image: url("{arrow_right}"); width: 8px; height: 8px; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QProgressBar {{
        background: {field}; color: {text}; border: 2px solid; border-color: {field_sunken};
        border-radius: 0px; text-align: center;
    }}
    QProgressBar::chunk {{ background: {navy}; width: 8px; margin: 1px; }}
    QSlider::groove:horizontal {{ height: 4px; background: {field}; border: 1px solid; border-color: {sunken}; border-radius: 0px; }}
    QSlider::handle:horizontal {{
        width: 10px; margin: -7px 0; background: {face}; border: 2px solid; border-color: {raised}; border-radius: 0px;
    }}

    QLabel#hoverReadout {{ background: {field}; border: 2px solid; border-color: {field_sunken}; border-radius: 0px; }}
    QToolButton#sectionHeader {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised};
        border-radius: 0px; padding: 4px 8px; font-weight: bold;
    }}
    QToolButton#sectionHeader:hover, QToolButton#sectionHeader:checked {{
        background: {face}; color: {text}; border: 2px solid; border-color: {raised};
    }}
    QWidget#sectionBody {{
        background: {face}; border: 2px groove {shadow}; border-top: none; border-radius: 0px;
    }}
    QFrame#plotToolbar, QFrame#datasetOpsPanel {{
        background: {face}; border: 1px solid; border-color: {raised}; border-radius: 0px;
    }}
    QWidget#featurePlacementUnitsBar, QLabel#featureWorkflowSteps, QLabel#featureSummary,
    QLabel#featureBuildSummary, QLabel#featureAssemblyStatus, QLabel#featureContract,
    QLabel#featureSurfaceBindingStatus, QGroupBox#featureStepCard {{
        background: {face}; border-radius: 0px;
    }}
    QLabel#featureEffectivePhysics, QLabel#featureModelBoundary, QLabel#featureValidationWarning {{
        background: {field}; border-radius: 0px;
    }}
    QWidget#plotSettingsContent, QWidget#featureAssemblyContent, QWidget#ghostSolverControlsContent,
    QWidget#pptControlsContent, QScrollArea#pptControlsScroll > QWidget,
    QScrollArea#featureBodyScroll, QScrollArea#featurePointScroll, QScrollArea#featureLineScroll,
    QScrollArea#featureReviewScroll, QScrollArea#plotSettingsScroll, QScrollArea#pptControlsScroll,
    QScrollArea#ghostSolverControlsScroll, QScrollArea#freddyWorkspaceScroll {{ background: {face}; }}
    """
