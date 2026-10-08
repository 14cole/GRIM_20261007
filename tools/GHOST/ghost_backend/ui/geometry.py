import math
import copy
import threading
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from ghost_backend.geometry.materials import (
    METERS_PER_INCH,
    segment_type_options,
    show_material_guide,
    choose_thin_layer,
)
from ghost_backend.ui.table_editors import ScrollSafeComboBox
from ghost_backend.ui.geometry_inspection import GeometryInspectionMixin

try:
    from PySide6.QtCore import Qt, Signal, QItemSelectionModel, QThread
    from PySide6.QtWidgets import (
        QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
        QHeaderView, QLabel, QMessageBox, QPushButton, QSizePolicy, QSplitter,
        QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    from PySide2.QtCore import Qt, Signal, QItemSelectionModel, QThread  # type: ignore
    from PySide2.QtWidgets import (  # type: ignore
        QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
        QHeaderView, QLabel, QMessageBox, QPushButton, QSizePolicy, QSplitter,
        QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from ghost_backend.geometry.io import (
    IBC_KINDS,
    AtomicFileTransaction,
    ChainSpec,
    Segment,
    build_geometry_snapshot,
    build_geometry_text,
    check_orientation_consistency,
    is_ibc_inline_row,
    is_tabulated_row,
    material_filename_from_row,
    parse_geometry,
)


SEGMENT_TYPE_OPTIONS: 'List[Tuple[str, str]]' = segment_type_options()
MESH_N_TOOLTIP = (
    "0 uses a nominal 20 panels per controlling wavelength (the same as -20), "
    "with sizing based on frequency, materials, and geometry. Positive N is "
    "a panel count per primitive; negative N is panels per wavelength. "
    "Certify mesh convergence separately checks accuracy and can refine "
    "supported 2D cases; 0 alone does not choose an optimal density."
)


def _parse_mesh_n_token(token: 'Any') -> 'int':
    """Parse the geometry N field without changing its solver semantics.

    Blank or zero selects automatic 20-panels-per-wavelength meshing, a
    positive integer is an explicit panel count per primitive, and a negative
    integer selects its absolute value in panels per wavelength.
    """

    text = str(token or "").strip()
    if not text:
        return 0
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("N must be an integer.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("N must be an integer.")
    return int(value)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=.16, right=.97, bottom=.16, top=.90)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


    def inaxes(self, xy):
        # Matplotlib's default canvas hit test omits child axes. Route events
        # before picking/navigation callbacks so the inset owns its clicks.
        main = super().inaxes(xy)
        for child in sorted(self.ax.child_axes, key=lambda ax: ax.get_zorder(), reverse=True):
            if child.get_visible() and child.patch.contains_point(xy):
                return child
        return main


class GeometryNavigationToolbar(NavigationToolbar):
    def press_pan(self, event):
        if event.inaxes in self.canvas.ax.child_axes:
            return
        super().press_pan(event)

    def press_zoom(self, event):
        if event.inaxes in self.canvas.ax.child_axes:
            return
        super().press_zoom(event)


class _GeometryValidationWorker(QThread):
    ready = Signal(int, object)
    failed = Signal(str)
    active = set()

    def __init__(self):
        super().__init__()
        self.abort = threading.Event()
        self.active.add(self)
        self.finished.connect(lambda: self.active.discard(self))

    def checkpoint(self):
        if self.abort.is_set():
            raise InterruptedError('Validation canceled.')

    def run(self):
        try:
            result = self.audit.run()
            self.checkpoint()
            self.ready.emit(self.version, result)
        except Exception as exc:
            self.failed.emit(str(exc))


class GeometryTab(GeometryInspectionMixin, QWidget):
    dirty_changed = Signal(bool)


    geometry_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._validation_version = 0
        self._validation_worker = None
        self.geometry_changed.connect(self._cancel_validation)
        splitter = QSplitter(Qt.Horizontal)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        self.canvas = MplCanvas(plot_container)

        self.toolbar = GeometryNavigationToolbar(self.canvas, plot_container)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("geometryStatus")
        self.lbl_status.setWordWrap(True)


        self.lbl_status.setStyleSheet("padding: 4px; font-family: monospace;")
        plot_layout.addWidget(self.lbl_status)
        splitter.addWidget(plot_container)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_validate = QPushButton("Validate")
        self.chk_show_normals = QCheckBox("Show Normals")
        self.chk_show_normals.setToolTip(
            "Show normals at a readable screen size, spaced to avoid crowding. "
            "Zoom in to reveal more arrows. Arrow = facing material; short "
            "rear tick = material behind. Select a segment for its side labels."
        )
        self.chk_show_impedance = QCheckBox("Show Impedance")
        self.chk_show_impedance.setToolTip(
            "Colour segments by IBC impedance. Tapered segments show a gradient; "
            "start = green dot, end = red dot."
        )
        self.chk_fill_materials = QCheckBox("Fill Materials")
        self.chk_fill_materials.setToolTip(
            "Fill enclosed regions with their material colour (grey = PEC, "
            "coloured tints = dielectrics, background = air) as implied by each "
            "segment's winding. A region whose boundary segments disagree "
            "about the enclosed material is hatched red."
        )
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_validate)
        btn_row.addStretch(1)
        right_layout.addLayout(btn_row)

        self.cmb_geometry_mode = ScrollSafeComboBox()
        self.cmb_geometry_mode.addItem("2D geometry", "2d")
        self.cmb_geometry_mode.addItem("BoR profile (X = radius)", "bor")
        self.cmb_geometry_mode.setToolTip(
            "Choose how to preview and validate this geometry. BoR revolves the "
            "X >= 0 half-profile around X = 0; this also selects the solver mode."
        )
        btn_row.insertWidget(2, self.cmb_geometry_mode)
        display_row = QHBoxLayout()
        self.cmb_normal_scope = ScrollSafeComboBox()
        self.cmb_normal_scope.addItem("Auto spaced", "auto")
        self.cmb_normal_scope.addItem("Selected only", "selected")
        self.cmb_normal_scope.addItem("All primitives", "all")
        self.cmb_normal_scope.setToolTip("Selected only isolates normals on the selected rows. All primitives can overlap.")
        for widget in (self.chk_show_normals, self.cmb_normal_scope,
                       self.chk_show_impedance, self.chk_fill_materials):
            display_row.addWidget(widget)
        display_row.addStretch(1)
        plot_layout.insertLayout(1, display_row)
        focus_row = QHBoxLayout()
        self.btn_fit_selected = QPushButton("Fit selected")
        self.btn_fit_all = QPushButton("Fit all")
        self.btn_fit_selected.setToolTip("Fit the selected segments; then scroll at a narrow gap to inspect it.")
        focus_row.addWidget(self.btn_fit_selected)
        focus_row.addWidget(self.btn_fit_all)
        self.lbl_preview_hint = QLabel("Scroll to zoom at cursor. Select a boundary to inspect its materials.")
        self.lbl_preview_hint.setWordWrap(True)
        focus_row.addWidget(self.lbl_preview_hint, 1)
        plot_layout.insertLayout(2, focus_row)

        self.btn_material_models = QPushButton("Material models and boundary sides...")
        self.btn_material_models.clicked.connect(lambda: show_material_guide(self))
        right_layout.addWidget(self.btn_material_models)

        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Type", "N (0=auto)", "IBC/Resistance",
             "pos_mat", "neg_mat"]
        )
        self.table.horizontalHeaderItem(2).setToolTip(MESH_N_TOOLTIP)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(self.table)

        self.validation_results = QTableWidget(0, 3)
        self.validation_results.setHorizontalHeaderLabels(["Level", "Row", "Finding (click to locate)"])
        self.validation_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.validation_results.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.validation_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.validation_results.setColumnWidth(0, 58)
        self.validation_results.setColumnWidth(1, 42)
        self.validation_results.setMaximumHeight(200)
        self.validation_results.setWordWrap(True)
        self.validation_results.hide()
        self.validation_results.cellClicked.connect(self._select_validation_finding)
        right_layout.addWidget(self.validation_results)

        mesh_actions = QHBoxLayout()
        self.btn_find_refinement = QPushButton("Find corners / junctions")
        self.btn_refine_selected = QPushButton("Refine selected 2x")
        self.btn_refine_selected.setToolTip("Increase N only on selected segments. Auto N becomes 40 panels per material wavelength. Fine geometry can already be denser; mesh certification still compares realized meshes.")
        self.btn_reverse_selected = QPushButton("Reverse selected")
        self.btn_reverse_selected.setToolTip(
            "Reverse the selected segments so their normals point the other way. "
            "Material assignments stay the same; start/end impedance tapers "
            "follow the new direction. Repeat to restore the original direction."
        )
        mesh_actions.addWidget(self.btn_find_refinement)
        mesh_actions.addWidget(self.btn_refine_selected)
        mesh_actions.addWidget(self.btn_reverse_selected)
        right_layout.addLayout(mesh_actions)
        self.btn_find_refinement.clicked.connect(self._select_refinement_candidates)
        self.btn_refine_selected.clicked.connect(self._refine_selected_segments)
        self.btn_reverse_selected.clicked.connect(self._reverse_selected_segments)

        bottom_row = QHBoxLayout()

        ibc_box = QVBoxLayout()
        self.lbl_ibc = QLabel("IBCS/Resistances")
        self.table_ibc = QTableWidget()
        self.table_ibc.setRowCount(0)
        self.table_ibc.setColumnCount(0)
        ibc_box.addWidget(self.lbl_ibc)
        ibc_box.addWidget(self.table_ibc)
        ibc_btn_row = QHBoxLayout()
        self.btn_ibc_add = QPushButton("+")
        self.btn_ibc_add_csv = QPushButton("+ CSV")
        self.btn_ibc_add_csv.setToolTip("Comma-separated CSV with header: frequency_hz,resistance_ohm,reactance_ohm. Frequency in Hz; impedance in ohms.")
        self.btn_thin_layer_add = QPushButton("+ Thin layer")
        self.btn_ibc_remove = QPushButton("-")
        ibc_btn_row.addWidget(self.btn_ibc_add)
        ibc_btn_row.addWidget(self.btn_ibc_add_csv)
        ibc_btn_row.addWidget(self.btn_thin_layer_add)
        ibc_btn_row.addWidget(self.btn_ibc_remove)
        ibc_btn_row.addStretch(1)
        ibc_box.addLayout(ibc_btn_row)
        self.btn_apply_conductor_ibc = QPushButton("Apply IBC to selected TYPE 2 segments")
        self.btn_apply_conductor_ibc.setToolTip("Select an impedance material row and the conductor segments above. For a FREDDY PEC-backed stack, draw the outer coating envelope. This assigns its scalar IBC without adding bulk layers or offsetting geometry.")
        ibc_box.addWidget(self.btn_apply_conductor_ibc)

        diel_box = QVBoxLayout()
        self.lbl_diel = QLabel("Dielectrics")
        self.table_diel = QTableWidget()
        self.table_diel.setRowCount(0)
        self.table_diel.setColumnCount(0)
        diel_box.addWidget(self.lbl_diel)
        diel_box.addWidget(self.table_diel)
        diel_btn_row = QHBoxLayout()
        self.btn_diel_add = QPushButton("+")
        self.btn_diel_add_csv = QPushButton("+ CSV")
        self.btn_diel_add_csv.setToolTip("Comma-separated CSV with header: frequency_hz,eps_real,eps_imag,mu_real,mu_imag. Frequency in Hz; relative epsilon and mu.")
        self.btn_diel_remove = QPushButton("-")
        diel_btn_row.addWidget(self.btn_diel_add)
        diel_btn_row.addWidget(self.btn_diel_add_csv)
        diel_btn_row.addWidget(self.btn_diel_remove)
        diel_btn_row.addStretch(1)
        diel_box.addLayout(diel_btn_row)

        bottom_row.addLayout(ibc_box, stretch=1)
        bottom_row.addLayout(diel_box, stretch=1)
        right_layout.addLayout(bottom_row)

        splitter.addWidget(right_container)

        splitter.setSizes([700, 300])
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        self.btn_load.clicked.connect(self.load_geo)
        self.btn_save.clicked.connect(self.save_geo)
        self.btn_validate.clicked.connect(self.validate_geometry)
        self.btn_ibc_add.clicked.connect(self._ibc_add_row)
        self.btn_ibc_add_csv.clicked.connect(self._ibc_add_csv_row)
        self.btn_thin_layer_add.clicked.connect(self._thin_layer_add_row)
        self.btn_apply_conductor_ibc.clicked.connect(self._apply_selected_conductor_ibc)
        self.btn_ibc_remove.clicked.connect(self._ibc_remove_row)
        self.btn_diel_add.clicked.connect(self._diel_add_row)
        self.btn_diel_add_csv.clicked.connect(self._diel_add_csv_row)
        self.btn_diel_remove.clicked.connect(self._diel_remove_row)
        self.chk_show_normals.toggled.connect(self._on_show_normals_toggled)
        self.chk_show_impedance.toggled.connect(self._on_show_impedance_toggled)
        self.chk_fill_materials.toggled.connect(self._on_fill_materials_toggled)

        self.title: 'str' = "Geometry"
        self.segments: 'List[Segment]' = []
        self.ibcs_entries: 'List[List[str]]' = []
        self.dielectric_entries: 'List[List[str]]' = []
        self.segment_lines: 'List' = []
        self.segment_base_colors: 'List[str]' = []
        self._populating: 'bool' = False
        self._syncing_selection: 'bool' = False
        self._selected_row: 'Optional[int]' = None
        self._last_ext: 'str' = ".geo"
        self.loaded_path: 'str' = ""
        self._dirty: 'bool' = False
        self._plot_theme: 'Optional[Dict[str, str]]' = None
        self.issue_rows: 'Set[int]' = set()
        self.normal_artists: 'List[Any]' = []

        self.impedance_artists: 'List[Any]' = []

        self.fill_artists: 'List[Any]' = []

        self.table.itemChanged.connect(self._on_main_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_ibc.itemChanged.connect(self._on_small_table_changed)
        self.table_diel.itemChanged.connect(self._on_small_table_changed)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)

        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self._normal_view = None
        self._axis_artist = None
        self.canvas.mpl_connect("draw_event", self._on_preview_draw)
        self.cmb_geometry_mode.currentIndexChanged.connect(self._on_geometry_mode_changed)
        self.cmb_normal_scope.currentIndexChanged.connect(self._on_show_normals_toggled)
        self.btn_fit_selected.clicked.connect(lambda: self._fit_geometry(selected=True))
        self.btn_fit_all.clicked.connect(lambda: self._fit_geometry(selected=False))

        self._init_inspection(plot_layout)

        self._set_equal_column_widths(self.table, enabled=True)
        self._set_equal_column_widths(self.table_ibc, enabled=False)
        self._set_equal_column_widths(self.table_diel, enabled=False)

    def geometry_mode(self):
        return str(self.cmb_geometry_mode.currentData() or "2d")

    def set_geometry_mode(self, mode):
        index = self.cmb_geometry_mode.findData(str(mode).lower())
        if index < 0:
            raise ValueError("Geometry mode must be '2d' or 'bor'.")
        self.cmb_geometry_mode.setCurrentIndex(index)

    def _on_geometry_mode_changed(self, *_):
        self._cancel_validation()
        self._update_geometry_axes()
        self._render_fills()
        self._render_normals()
        self.lbl_status.setText("BoR profile: X is radius, Y is axial Z; validate against the axisymmetric solver rules."
                                if self.geometry_mode() == "bor" else "2D geometry: validate planar boundaries and materials.")
        self.canvas.draw_idle()

    def _update_geometry_axes(self):
        if self._axis_artist is not None:
            try:
                self._axis_artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._axis_artist = None
        ax = self.canvas.ax
        if self.geometry_mode() == "bor":
            ax.set_xlabel(f"Radius X ({self._geometry_unit_label()})")
            ax.set_ylabel(f"Axial Z = Y ({self._geometry_unit_label()})")
            self._axis_artist = ax.axvline(0, color="#888888", linestyle="--", linewidth=1, zorder=2)
        else:
            ax.set_xlabel(f"X ({self._geometry_unit_label()})")
            ax.set_ylabel(f"Y ({self._geometry_unit_label()})")

    def _fit_geometry(self, *, selected=False):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}) if selected else range(len(self.segments))
        points = [(x, y) for row in rows for x, y in zip(self.segments[row].x, self.segments[row].y)
                  if math.isfinite(x) and math.isfinite(y)]
        if not points:
            self.lbl_status.setText("Select one or more segments to fit." if selected else "Load geometry to preview.")
            return
        xs, ys = zip(*points)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        pad = 0.08 * span
        self.canvas.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.canvas.ax.set_ylim(min(ys) - pad, max(ys) + pad)
        self.canvas.draw_idle()

    def _preview_view_key(self):
        ax = self.canvas.ax
        return (*ax.get_xlim(), *ax.get_ylim(), *ax.bbox.bounds)

    def _on_preview_draw(self, _event):
        changed = False
        if self.chk_show_normals.isChecked() and self._normal_view != self._preview_view_key():
            self._render_normals()
            changed = True
        if hasattr(self, "chk_detail_inset") and self._refresh_detail_if_needed():
            changed = True
        if changed:
            self.canvas.draw_idle()

    def _select_validation_finding(self, row, _column):
        item = self.validation_results.item(row, 0)
        target = item.data(Qt.UserRole) if item else None
        if target is not None and 0 <= target < len(self.segments):
            if not self._inspection_row_matches(target):
                self.cmb_material_isolation.setCurrentIndex(0)
            self.table.selectRow(target)
            self._apply_selection(target)
            self._fit_geometry(selected=True)

    def is_dirty(self) -> 'bool':
        return bool(self._dirty)

    def _set_dirty(self, dirty: 'bool') -> 'None':
        value = bool(dirty)
        if value:
            self.geometry_changed.emit()
        if value == self._dirty:
            return
        self._dirty = value
        self.dirty_changed.emit(value)

    def _confirm_unsaved_changes(
        self, *, action: 'str', parent: 'Optional[QWidget]' = None
    ) -> 'bool':
        if not self.is_dirty():
            return True
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        shown_name = Path(self.loaded_path).name if self.loaded_path else "Untitled geometry"
        answer = QMessageBox.warning(
            parent or self,
            "Unsaved GHOST Geometry",
            f"'{shown_name}' has unsaved geometry or material changes. "
            f"Save them before {action}?",
            buttons.Save | buttons.Discard | buttons.Cancel,
            buttons.Save,
        )
        if answer == buttons.Cancel:
            return False
        if answer == buttons.Save:
            return bool(self.save_geo())
        return True

    def request_close(self, parent: 'Optional[QWidget]' = None) -> 'bool':
        if self._gap_worker is not None and self._gap_worker.isRunning():
            self._clear_gap()
            self.lbl_status.setText('Canceling minimum gap search. Close again after it finishes.')
            return False
        if self._validation_worker is not None and self._validation_worker.isRunning():
            self._cancel_validation()
            self.lbl_status.setText('Canceling validation. Close again after it finishes.')
            return False
        return self._confirm_unsaved_changes(action="closing GHOST", parent=parent)

    def apply_plot_theme(
        self, *, background: 'str', text: 'str', grid: 'str'
    ) -> 'None':
        """Apply the embedding shell's colors without changing standalone defaults."""

        self._plot_theme = {
            "background": str(background),
            "text": str(text),
            "grid": str(grid),
        }
        self._apply_plot_theme_to_axes()
        self._render_fills()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _apply_plot_theme_to_axes(self) -> 'None':
        if self._plot_theme is None:
            return
        background = self._plot_theme["background"]
        text = self._plot_theme["text"]
        grid = self._plot_theme["grid"]
        self.canvas.fig.patch.set_facecolor(background)
        ax = self.canvas.ax
        ax.set_facecolor(background)
        ax.title.set_color(text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.tick_params(axis="both", colors=text)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(True, color=grid, alpha=0.45)

    def _segment_plot_colors(self) -> 'List[str]':
        dark_text = self._plot_theme["text"] if self._plot_theme else "black"
        return ["orange", "green", "#60a5fa", "gray", dark_text, "red", "#c084fc", "cyan"]

    def load_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return False
        if not self._confirm_unsaved_changes(action="loading another geometry"):
            return False
        try:
            with open(fname, "r") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
            return False
        try:
            title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse geometry: {e}")
            return False

        self.title = title
        self.segments = segments
        self.ibcs_entries = ibcs_entries
        self.dielectric_entries = dielectric_entries
        self.loaded_path = os.path.abspath(fname)

        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self.segments))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(
                ["Name", "Type", "N (0=auto)", "IBC/Resistance",
                 "pos_mat", "neg_mat"]
            )
            self.table.horizontalHeaderItem(2).setToolTip(MESH_N_TOOLTIP)
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 5)
                n_value = props[1] if len(props) >= 2 else ""
                self.table.setItem(row, 0, QTableWidgetItem(seg.name))
                self.table.setItem(row, 2, QTableWidgetItem(n_value))
                self.table.item(row, 2).setToolTip(MESH_N_TOOLTIP)
        finally:
            self._populating = False

        self._inspection_before_load()
        ax = self.canvas.ax
        ax.clear()
        self.segment_lines = []
        self.segment_base_colors = []
        self.issue_rows.clear()
        self._clear_normals()

        plot_colors = self._segment_plot_colors()

        for row, seg in enumerate(self.segments):
            props = seg.properties
            itype = props[0] if len(props) >= 1 else ""
            try:
                color_index = (int(itype) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except (ValueError, TypeError):
                base_color = plot_colors[row % len(plot_colors)]

            plot_x, plot_y = self._segment_plot_xy(seg)
            (line2d,) = ax.plot(plot_x, plot_y, color=base_color, linewidth=1.5, zorder=1)
            line2d.set_picker(True)
            line2d.set_pickradius(5)
            self.segment_lines.append(line2d)
            self.segment_base_colors.append(base_color)
        ax.set_title(self.title)
        self._update_geometry_axes()
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        self._apply_plot_theme_to_axes()

        self._populate_small_table(self.table_ibc, self.ibcs_entries, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._populate_small_table(
            self.table_diel,
            self.dielectric_entries,
            label=self.lbl_diel,
            title_prefix="Dielectrics",
        )
        self._refresh_segment_dropdowns()

        self._selected_row = None
        self._refresh_segment_styles()
        self._render_normals()
        self._render_impedance_overlay()
        self._render_fills()
        self._update_status_label(-1)
        self.canvas.draw()


        self.geometry_changed.emit()
        self._set_dirty(False)
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.segments)} segments(s),"
            f"{len(self.ibcs_entries)} IBCS/Resistances entry(ies),"
            f"and {len(self.dielectric_entries)} dielectric entry(ies).",
        )
        return True

    def _populate_small_table(self, table: 'QTableWidget', rows: 'List[List[str]]', label: 'QLabel', title_prefix: 'str'):
        if title_prefix == "IBCS/Resistances":


            headers = ["Flag", "Model / CSV", "R / d (in)", "X / material", "R_end", "X_end"]
            col_count = len(headers)
        elif title_prefix == "Dielectrics":
            col_count = 5
            headers = ["Flag", "CSV file / Ep_real", "Ep_imag", "Mu_real", "Mu_imag"]
        else:
            col_count = max((len(r) for r in rows), default=0)
            headers = [f"Col {i+1}" for i in range(col_count)]

        table.blockSignals(True)
        try:


            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    if table.cellWidget(r, c) is not None:
                        table.removeCellWidget(r, c)
            table.clearContents()
            table.setRowCount(len(rows))
            table.setColumnCount(col_count)
            table.setHorizontalHeaderLabels(headers)
            widths = [44, 135, 88, 88, 70, 70] if title_prefix == "IBCS/Resistances" else [44, 130, 75, 75, 75]
            for col in range(col_count):
                table.setColumnWidth(col, widths[col] if col < len(widths) else 85)
                table.horizontalHeaderItem(col).setToolTip(headers[col])

            for r, tokens in enumerate(rows):
                for c, token in enumerate(tokens):
                    if c >= col_count:
                        break
                    table.setItem(r, c, QTableWidgetItem(token))


            if title_prefix == "IBCS/Resistances":
                for r, tokens in enumerate(rows):
                    if len(tokens) < 2 or is_tabulated_row(tokens):
                        continue
                    if str(tokens[1]).lower() == "thin_dielectric":
                        item = table.item(r, 2)
                        try:
                            shown = format(float(tokens[2]) / METERS_PER_INCH, ".12g")
                        except (ValueError, OverflowError):
                            shown = tokens[2]
                        item.setText(shown)
                        table.setColumnWidth(2, max(table.columnWidth(2), table.fontMetrics().horizontalAdvance(shown) + 16))


                        item.setData(Qt.UserRole, (shown, tokens[2]))
                        item.setToolTip("Physical layer thickness in inches")
                        table.item(r, 3).setToolTip("Flag in the Dielectrics table (epsilon and mu)")
                        continue
                    self._install_ibc_kind_combo(table, r, tokens[1])
        finally:
            table.blockSignals(False)

        label.setText(f"{'Surface materials (IBC / sheets)' if title_prefix == 'IBCS/Resistances' else title_prefix} (n={len(rows)})")

    def _install_ibc_kind_combo(self, table: 'QTableWidget', row: 'int', current_kind: 'str') -> 'None':
        cb = ScrollSafeComboBox()
        for kind in IBC_KINDS:
            cb.addItem(kind, userData=kind)
        target = (current_kind or "").strip().lower()
        idx = next((i for i, k in enumerate(IBC_KINDS) if k == target), 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(self._on_small_table_changed)


        table.setItem(row, 1, None)
        table.setCellWidget(row, 1, cb)

    def _ensure_prop_len(self, props: 'List[str]', n: 'int') -> 'List[str]':
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props

    def _ibc_dropdown_options(self) -> 'List[Tuple[str, str]]':
        """Build (value, label) pairs for the IBC dropdown from current IBC table state."""
        options: 'List[Tuple[str, str]]' = [("0", "0 (none)")]
        lookup = self._ibcs_lookup()
        for flag in sorted(lookup.keys()):
            kind = lookup[flag].get("kind", "undefined")
            options.append((str(flag), f"{flag} ({kind})"))
        return options

    def _diel_dropdown_options(self) -> 'List[Tuple[str, str]]':
        """Build (value, label) pairs for the pos_mat/neg_mat dropdown from current dielectric table state."""
        options: 'List[Tuple[str, str]]' = [("0", "0 (vacuum)")]
        seen: 'Set[int]' = set()
        rows = self._read_small_table(self.table_diel)
        for row in rows:
            if not row:
                continue
            try:
                flag = int(row[0])
            except (ValueError, TypeError):
                continue
            if flag <= 0 or flag in seen:
                continue
            seen.add(flag)
            if is_tabulated_row(row):
                filename = material_filename_from_row(row)
                options.append((str(flag), f"{flag} ({filename})"))
            else:
                options.append((str(flag), f"{flag} (constant)"))
        options.sort(key=lambda x: int(x[0]))
        return options

    def _make_segment_combo(
        self,
        options: 'List[Tuple[str, str]]',
        current: 'str',
        row: 'int',
        prop_index: 'int',
    ) -> 'QComboBox':
        """Build a QComboBox for the segment table. Connects the change handler only after the initial index is set."""
        cb = ScrollSafeComboBox()
        found_index = -1
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(8)
        cb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        current_clean = (current or "").strip()
        for i, (value, lbl) in enumerate(options):
            cb.addItem(lbl, userData=value)
            if value == current_clean:
                found_index = i
        if found_index < 0 and current_clean:
            cb.addItem(f"{current_clean} (undefined)", userData=current_clean)
            found_index = cb.count() - 1
        if found_index >= 0:
            cb.setCurrentIndex(found_index)
        cb.currentIndexChanged.connect(
            lambda _idx, r=row, p=prop_index, w=cb: self._on_segment_combo_changed(r, p, w)
        )
        return cb

    def _refresh_segment_dropdowns(self) -> 'None':
        """Rebuild all segment-table comboboxes against the current IBC/dielectric tables."""
        if not self.segments:
            return
        ibc_opts = self._ibc_dropdown_options()
        diel_opts = self._diel_dropdown_options()
        was_populating = self._populating
        self._populating = True
        try:
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 5)
                type_val = (props[0] or "").strip()
                self.table.setCellWidget(row, 1, self._make_segment_combo(SEGMENT_TYPE_OPTIONS, type_val, row, 0))
                self.table.setCellWidget(row, 3, self._make_segment_combo(ibc_opts, (props[2] or "").strip(), row, 2))
                self.table.setCellWidget(row, 4, self._make_segment_combo(diel_opts, (props[3] or "").strip(), row, 3))
                self.table.setCellWidget(row, 5, self._make_segment_combo(diel_opts, (props[4] or "").strip(), row, 4))
                self._apply_neg_mat_editability(row, type_val)
        finally:
            self._populating = was_populating

    def _on_segment_combo_changed(self, row: 'int', prop_index: 'int', combo: 'QComboBox') -> 'None':
        if self._populating:
            return
        if row < 0 or row >= len(self.segments):
            return
        new_value = combo.currentData()
        if new_value is None:
            new_value = combo.currentText()
        new_value = str(new_value)

        seg = self.segments[row]
        props = self._ensure_prop_len(seg.properties, 5)
        props[prop_index] = new_value
        self._set_dirty(True)

        if prop_index == 0:
            seg.seg_type = new_value or None
            plot_colors = self._segment_plot_colors()
            try:
                color_index = (int(new_value) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except (ValueError, TypeError):
                base_color = plot_colors[row % len(plot_colors)]
            if row < len(self.segment_base_colors):
                self.segment_base_colors[row] = base_color
            self._refresh_segment_styles()
            self._apply_neg_mat_editability(row, new_value)

        if prop_index in (0, 2):
            self._render_impedance_overlay()
            self.canvas.draw_idle()
        if prop_index in (0, 3, 4):
            self._render_normals()
            self._render_fills()
            self.canvas.draw_idle()

        if row == self._selected_row:
            self._update_status_label(row)

    def _ibc_add_row(self) -> 'None':
        current = self._read_small_table(self.table_ibc) if self.table_ibc.rowCount() > 0 else []
        used: 'Set[int]' = set()
        for row in current:
            try:
                used.add(int(row[0]))
            except (ValueError, IndexError, TypeError):
                pass
        next_flag = 1
        while next_flag in used:
            next_flag += 1

        current.append([str(next_flag), "constant", "50", "0", "0", "0"])
        self.ibcs_entries = current
        self._populate_small_table(self.table_ibc, current, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _thin_layer_add_row(self):
        options = [(flag, label) for flag, label in self._diel_dropdown_options() if flag != "0"]
        if not options:
            QMessageBox.information(self, "Thin layer material", "Add a dielectric material (epsilon and mu) first.")
            return
        row = choose_thin_layer(self, options)
        if row is None:
            return
        current = self._read_small_table(self.table_ibc)
        used = {self._parse_int_token(item[0], 0) for item in current if item}
        flag = 1
        while flag in used:
            flag += 1
        current.append([str(flag)] + row)
        self.ibcs_entries = current
        self._populate_small_table(self.table_ibc, current, self.lbl_ibc, "IBCS/Resistances")
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _choose_material_csv(self, title: 'str', *, kind: 'Optional[str]' = None) -> 'str':
        if not self.loaded_path:
            QMessageBox.warning(
                self,
                "Save Geometry First",
                "Save or load the geometry before adding a material CSV. "
                "Material files must be beside the .geo file.",
            )
            return ""
        base_dir = os.path.dirname(os.path.abspath(self.loaded_path))
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self, title, base_dir, "CSV Files (*.csv)"
        )
        if not filename:
            return ""
        if os.path.dirname(os.path.abspath(filename)) != base_dir:
            QMessageBox.warning(
                self,
                "Material File Location",
                "Choose a CSV in the same directory as the geometry file:\n"
                f"{base_dir}",
            )
            return ""
        basename = os.path.basename(filename)
        try:
            validated_name = material_filename_from_row(["1", basename])
            if validated_name is None:
                raise ValueError(f"Unsupported material filename: {basename!r}")
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Unsupported Material Filename",
                f"This CSV cannot be referenced by a .geo file:\n{exc}",
            )
            return ""
        if kind is not None:
            try:
                from ghost_backend.twod.geometry import MaterialLibrary
                rows = [["1", validated_name]]
                MaterialLibrary.from_entries(
                    rows if kind == "ibc" else [],
                    rows if kind == "material" else [],
                    base_dir,
                )
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Invalid Material CSV", str(exc))
                return ""
        return validated_name

    def attach_material_artifact(
        self, artifact_kind: 'str', csv_path: 'str'
    ) -> 'bool':
        """Copy a typed FREDDY artifact beside the active ``.geo`` and add it.

        This is intentionally a material-table handoff, not a generic CSV
        import. The exact production solver reader validates the source before
        a file is copied or a geometry row is changed.
        """

        kind = str(artifact_kind).strip().lower()
        if kind not in {"ibc", "material"}:
            QMessageBox.warning(
                self,
                "Unsupported FREDDY Artifact",
                "GHOST accepts only nominal IBC or material artifacts from "
                "FREDDY. Analysis CSVs cannot be attached.",
            )
            return False

        geometry_path = Path(self.loaded_path).expanduser().resolve() \
            if self.loaded_path else None
        if (
            geometry_path is None
            or geometry_path.suffix.lower() != ".geo"
            or not geometry_path.is_file()
        ):
            QMessageBox.warning(
                self,
                "No Active Saved Geometry",
                "Load or save the current GHOST geometry as a .geo file before "
                "attaching a FREDDY artifact. The CSV must live beside that "
                "saved geometry.",
            )
            return False

        source = Path(csv_path).expanduser().resolve()
        if source.suffix.lower() != ".csv" or not source.is_file():
            QMessageBox.warning(
                self,
                "FREDDY Artifact Missing",
                f"The selected nominal artifact is not a readable CSV:\n{source}",
            )
            return False


        try:
            destination_name = material_filename_from_row(
                ["1", source.name]
            )
            if destination_name is None:
                raise ValueError(f"Unsupported material filename: {source.name!r}")
        except ValueError as exc:
            QMessageBox.critical(
                self,
                "Invalid FREDDY Artifact Filename",
                f"The nominal CSV cannot be referenced by a .geo file:\n{exc}",
            )
            return False


        try:
            from ghost_backend.twod.solver import MaterialLibrary

            if kind == "ibc":
                MaterialLibrary.from_entries(
                    [["1", source.name]], [], str(source.parent)
                )
            else:
                MaterialLibrary.from_entries(
                    [], [["1", source.name]], str(source.parent)
                )
        except Exception as exc:
            label = "IBC" if kind == "ibc" else "material"
            QMessageBox.critical(
                self,
                "Invalid FREDDY Artifact",
                f"The nominal {label} CSV is not compatible with GHOST:\n{exc}",
            )
            return False

        destination = geometry_path.parent / destination_name
        opposite_table = self.table_diel if kind == "ibc" else self.table_ibc
        opposite_kind = "dielectric material" if kind == "ibc" else "IBC"
        opposite_reference = next(
            (
                row
                for row in self._read_small_table(opposite_table)
                if len(row) == 2
                and str(row[1]).strip().casefold()
                == destination.name.casefold()
            ),
            None,
        )
        if opposite_reference is not None:
            QMessageBox.warning(
                self,
                "Geometry Sidecar Type Conflict",
                f"'{destination.name}' is already referenced by this geometry "
                f"as {opposite_kind} flag {opposite_reference[0]}. One CSV "
                "cannot use both material schemas. Export the FREDDY artifact "
                "under a different filename and try again.",
            )
            return False

        same_file = os.path.normcase(str(source)) == os.path.normcase(
            str(destination.resolve())
        )
        if destination.exists() and not same_file:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            answer = QMessageBox.question(
                self,
                "Replace Existing Geometry Sidecar?",
                f"A file named '{destination.name}' already exists beside:\n"
                f"{geometry_path.name}\n\n"
                "Replace it with the selected nominal FREDDY artifact? "
                "This can affect any geometry that references the same file.",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                return False

        if kind == "ibc":
            table = self.table_ibc
            previous_model = [list(row) for row in self.ibcs_entries]
            label = self.lbl_ibc
            title_prefix = "IBCS/Resistances"
            friendly_kind = "IBC"
        else:
            table = self.table_diel
            previous_model = [list(row) for row in self.dielectric_entries]
            label = self.lbl_diel
            title_prefix = "Dielectrics"
            friendly_kind = "dielectric material"

        previous_rows = [list(row) for row in self._read_small_table(table)]
        current = [list(row) for row in previous_rows]

        existing_row = next(
            (
                index
                for index, row in enumerate(current)
                if len(row) == 2
                and str(row[1]).strip().casefold()
                == destination.name.casefold()
            ),
            None,
        )
        if existing_row is None:
            used = {
                self._parse_int_token(row[0], 0)
                for row in current
                if row
            }
            flag = 1
            while flag in used:
                flag += 1
            current.append([str(flag), destination.name])
            selected_row = len(current) - 1
        else:
            selected_row = existing_row
            flag = self._parse_int_token(current[existing_row][0], 0)


            current[existing_row][1] = destination.name

        transaction = AtomicFileTransaction()
        try:
            if not same_file:
                transaction.stage_copy(source, destination)
            transaction.publish()

            if kind == "ibc":
                self.ibcs_entries = current
            else:
                self.dielectric_entries = current
            self._populate_small_table(
                table, current, label=label, title_prefix=title_prefix
            )
            table.selectRow(selected_row)
            self._refresh_segment_dropdowns()
            if kind == "ibc":
                self._render_impedance_overlay()
                self.canvas.draw_idle()
        except Exception as exc:
            rollback_errors: 'List[str]' = []


            if kind == "ibc":
                self.ibcs_entries = previous_model
            else:
                self.dielectric_entries = previous_model
            try:
                self._populate_small_table(
                    table,
                    previous_rows,
                    label=label,
                    title_prefix=title_prefix,
                )
            except Exception as rollback_exc:
                rollback_errors.append(f"table restore: {rollback_exc}")
            try:
                self._refresh_segment_dropdowns()
            except Exception as rollback_exc:
                rollback_errors.append(f"segment controls: {rollback_exc}")
            if kind == "ibc":
                try:
                    self._render_impedance_overlay()
                    self.canvas.draw_idle()
                except Exception as rollback_exc:
                    rollback_errors.append(f"preview restore: {rollback_exc}")
            try:
                transaction.abort()
            except Exception as rollback_exc:
                rollback_errors.append(f"file restore: {rollback_exc}")

            detail = f"Could not attach the nominal CSV:\n{exc}"
            if rollback_errors:
                detail += "\n\nRollback warning(s):\n" + "\n".join(
                    rollback_errors
                )
            else:
                detail += "\n\nThe previous file and geometry table were restored."
            QMessageBox.critical(
                self,
                "FREDDY Artifact Attachment Failed",
                detail,
            )
            return False

        transaction.commit()
        self._set_dirty(True)

        QMessageBox.information(
            self,
            "FREDDY Artifact Attached",
            f"Attached '{destination.name}' beside '{geometry_path.name}' as "
            f"{friendly_kind} flag {flag}.\n\n"
            + ("For a collapsed PEC-backed stack, select the TYPE 2 outer-envelope "
               "segments and use Apply IBC to selected TYPE 2 segments. "
               "Do not also model the collapsed bulk layers.\n\n" if kind == "ibc" else "")
            + "Use Save in the Geometry tab to persist this new reference in "
            "the .geo file.",
        )
        return True

    def _apply_selected_conductor_ibc(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        material_rows = sorted({index.row() for index in self.table_ibc.selectedIndexes()})
        try:
            if not rows or len(material_rows) != 1:
                raise ValueError("Select conductor segments above and exactly one surface-material row below.")
            definition = self._read_small_table(self.table_ibc)[material_rows[0]]
            flag = int(definition[0])
            from ghost_backend.twod.solver import MaterialLibrary
            base_dir = str(Path(self.loaded_path).parent) if self.loaded_path else os.getcwd()
            library = MaterialLibrary.from_entries([definition], [], base_dir)
            model = library.impedance_models[flag]
            frequency = float(model.freqs_ghz[0]) if hasattr(model, "freqs_ghz") else 1.
            library.get_impedance(flag, frequency)
            for row in rows:
                props = self._ensure_prop_len(self.segments[row].properties, 5)
                if int(props[0]) != 2 or any(int(props[i] or 0) != 0 for i in (3,4)):
                    raise ValueError("This action requires TYPE 2 conductor segments with air (0) region flags. Bulk interfaces and free sheets need their own material workflow.")
        except (ValueError, IndexError, KeyError, OSError) as exc:
            QMessageBox.warning(self, "Apply conductor IBC", str(exc))
            return False
        for row in rows:
            self.segments[row].properties[2] = str(flag)
        self._refresh_segment_dropdowns()
        self._render_impedance_overlay()
        self.canvas.draw_idle()
        self._set_dirty(True)
        self.lbl_status.setText(f"Applied IBC flag {flag} to {len(rows)} TYPE 2 segments. For a FREDDY coating, these coordinates must represent the outer coating envelope. Save the geometry to retain the assignment.")
        return True

    def _ibc_add_csv_row(self) -> 'None':
        filename = self._choose_material_csv("Choose IBC Material CSV (Hz)", kind="ibc")
        if not filename:
            return
        current = self._read_small_table(self.table_ibc)
        used = {
            self._parse_int_token(row[0], 0) for row in current if row
        }
        next_flag = 1
        while next_flag in used:
            next_flag += 1
        current.append([str(next_flag), filename])
        self.ibcs_entries = current
        self._populate_small_table(
            self.table_ibc, current, label=self.lbl_ibc,
            title_prefix="IBCS/Resistances"
        )
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _ibc_remove_row(self) -> 'None':
        sel = sorted({i.row() for i in self.table_ibc.selectedIndexes()}, reverse=True)
        if not sel:
            if self.table_ibc.rowCount() == 0:
                return
            sel = [self.table_ibc.rowCount() - 1]
        current = self._read_small_table(self.table_ibc)
        for r in sel:
            if 0 <= r < len(current):
                del current[r]
        self.ibcs_entries = current
        self._populate_small_table(self.table_ibc, current, label=self.lbl_ibc, title_prefix="IBCS/Resistances")
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _diel_add_row(self) -> 'None':
        current = self._read_small_table(self.table_diel) if self.table_diel.rowCount() > 0 else []
        used: 'Set[int]' = set()
        for row in current:
            try:
                used.add(int(row[0]))
            except (ValueError, IndexError, TypeError):
                pass
        next_flag = 1
        while next_flag in used:
            next_flag += 1

        current.append([str(next_flag), "1.0", "0.0", "1.0", "0.0"])
        self.dielectric_entries = current
        self._populate_small_table(self.table_diel, current, label=self.lbl_diel, title_prefix="Dielectrics")
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _diel_add_csv_row(self) -> 'None':
        filename = self._choose_material_csv("Choose Dielectric Material CSV (Hz)", kind="material")
        if not filename:
            return
        current = self._read_small_table(self.table_diel)
        used = {
            self._parse_int_token(row[0], 0) for row in current if row
        }
        next_flag = 1
        while next_flag in used:
            next_flag += 1
        current.append([str(next_flag), filename])
        self.dielectric_entries = current
        self._populate_small_table(
            self.table_diel, current, label=self.lbl_diel,
            title_prefix="Dielectrics"
        )
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _diel_remove_row(self) -> 'None':
        sel = sorted({i.row() for i in self.table_diel.selectedIndexes()}, reverse=True)
        if not sel:
            if self.table_diel.rowCount() == 0:
                return
            sel = [self.table_diel.rowCount() - 1]
        current = self._read_small_table(self.table_diel)
        for r in sel:
            if 0 <= r < len(current):
                del current[r]
        self.dielectric_entries = current
        self._populate_small_table(self.table_diel, current, label=self.lbl_diel, title_prefix="Dielectrics")
        self._refresh_segment_dropdowns()
        self._set_dirty(True)

    def _on_small_table_changed(self, *_args) -> 'None':
        if self._populating:
            return


        self._set_dirty(True)
        self._refresh_segment_dropdowns()

    def _apply_neg_mat_editability(self, row: 'int', seg_type: 'str') -> 'None':

        widget = self.table.cellWidget(row, 5)
        if widget is None:
            return
        widget.setEnabled(str(seg_type).strip() == "5")

    def _on_main_table_item_changed(self, item: 'QTableWidgetItem'):
        if self._populating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.segments):
            return

        seg = self.segments[row]
        text = item.text().strip()


        if col == 0:
            seg.name = text
        elif col == 2:
            props = self._ensure_prop_len(seg.properties, 5)
            props[1] = text
        self._set_dirty(True)

        if row == self._selected_row:
            self._update_status_label(row)

    def _select_refinement_candidates(self):
        from ghost_backend.geometry.guidance import geometry_refinement_candidates
        candidates = geometry_refinement_candidates(self.get_geometry_snapshot()["segments"])
        self.table.clearSelection()
        for row in candidates:
            self.table.selectionModel().select(self.table.model().index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.lbl_status.setText(f"Selected {len(candidates)} segments at open ends, bends, or material junctions. These are geometry indicators; use mesh certification to check complex-field accuracy.")

    def _refine_selected_segments(self):
        from ghost_backend.geometry.guidance import refined_density
        selected = sorted({index.row() for index in self.table.selectedIndexes()})
        try:
            values = [(row, refined_density(self._ensure_prop_len(self.segments[row].properties, 5)[1])) for row in selected]
        except ValueError as exc:
            QMessageBox.warning(self, "Mesh density", str(exc))
            return
        for row, value in values:
            self.table.item(row, 2).setText(value)
        self.lbl_status.setText(f"Refined {len(values)} selected segments. Run the certified solve to measure the field change." if values else "Select segments in the table or use Find corners / junctions first.")

    def _reverse_selected_segments(self):
        selected = sorted({index.row() for index in self.table.selectedIndexes()})
        if not selected:
            self.lbl_status.setText("Select segments in the table or preview to reverse their direction.")
            return

        updates = []
        for row in selected:
            seg = self.segments[row]
            if len(seg.x) != len(seg.y) or len(seg.x) % 2:
                QMessageBox.warning(
                    self, "Reverse segment direction",
                    f"Segment {seg.name} has incomplete coordinate pairs. "
                    "No segments were changed.",
                )
                return
            # Each consecutive pair is a directed line primitive. Reversing
            # both arrays reverses the chain and each primitive's normal.
            updates.append((row, list(reversed(seg.x)), list(reversed(seg.y))))

        lookup = self._ibcs_lookup()
        has_taper = False
        for row in selected:
            props = self._ensure_prop_len(self.segments[row].properties, 5)
            info = lookup.get(self._parse_int_token(props[2], 0), {})
            if (info.get("kind") in ("linear", "cosine", "exp")
                    and info.get("z_start") != info.get("z_end")):
                has_taper = True
        for row, xs, ys in updates:
            seg = self.segments[row]
            seg.x, seg.y = xs, ys
            if row < len(self.segment_lines):
                self.segment_lines[row].set_data(*self._segment_plot_xy(seg))

        self.issue_rows.clear()
        self._set_dirty(True)
        self._refresh_segment_styles()
        self._render_normals()
        self._render_impedance_overlay()
        self._render_fills()
        self.canvas.draw_idle()
        message = (
            f"Reversed {len(updates)} selected segment(s); normals now point the other way. "
            "Repeat to restore the original direction."
        )
        if has_taper:
            message += " Start/end impedance tapers now follow the reversed direction."
        self.lbl_status.setText(message)

    def _on_table_selection_changed(self):
        if self._syncing_selection:
            return
        selected = {index.row() for index in self.table.selectedIndexes()}
        row = self.table.currentRow()
        if row not in selected:
            row = min(selected) if selected else -1
        self._apply_selection(row)

    def _apply_selection(self, row: 'int'):
        self._selected_row = row if (row is not None and row >= 0) else None
        self._refresh_segment_styles()
        self._update_status_label(row if row is not None else -1)
        self._render_normals()
        self._inspection_selection_changed()
        self.canvas.draw_idle()

    def _on_plot_pick(self, event):
        if self.btn_measure_gap.isChecked() or (self._detail_ax is not None and self._inspection_axes(event.mouseevent) is self._detail_ax):
            return
        line = getattr(event, "artist", None)
        if not line:
            return
        if self.toolbar.mode:
            return
        row = self._hit_test(event.mouseevent)
        if row is None:
            return
        self._syncing_selection = True
        try:
            self.table.selectRow(row)
            self._apply_selection(row)
        finally:
            self._syncing_selection = False

    def _on_plot_button_press(self, event):
        if self._inspection_button_press(event):
            return
        if event.inaxes != self.canvas.ax or self.toolbar.mode:
            return
        modifier_select = event.button == 1 and (event.key in ("control", "shift"))
        if event.button == 3 or modifier_select:
            idx = self._hit_test(event)
            if idx is not None:
                self._syncing_selection = True
                try:
                    self.table.selectRow(idx)
                    self._apply_selection(idx)
                finally:
                    self._syncing_selection = False

    def _hit_test(self, event) -> 'Optional[int]':
        hit = self._nearest_primitive(event)
        return hit["row"] if hit is not None else None

    def _on_plot_scroll(self, event):
        if self._detail_ax is not None and self._inspection_axes(event) is self._detail_ax:
            step = 1 if event.button == "up" else -1
            index = max(0, min(self.cmb_detail_zoom.count() - 1, self.cmb_detail_zoom.currentIndex() + step))
            self.cmb_detail_zoom.setCurrentIndex(index)
            return
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2 if event.button == "up" else (1 / 1.2)
        self._zoom_at(event.xdata, event.ydata, base_scale)

    def _zoom_at(self, x: 'float', y: 'float', scale: 'float'):
        ax = self.canvas.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = (xlim[1] - xlim[0]) / scale
        h = (ylim[1] - ylim[0]) / scale
        fx = (x - xlim[0]) / (xlim[1] - xlim[0])
        fy = (y - ylim[0]) / (ylim[1] - ylim[0])
        ax.set_xlim(x - fx * w, x + (1 - fx) * w)
        ax.set_ylim(y - fy * h, y + (1 - fy) * h)
        self.canvas.draw_idle()

    def _refresh_segment_styles(self):
        selected = {index.row() for index in self.table.selectedIndexes()}
        for i, line in enumerate(self.segment_lines):
            if i in selected or (self._selected_row is not None and i == self._selected_row):
                line.set_color("pink")
                line.set_linewidth(2.5)
                line.set_zorder(10)
                continue
            if i in self.issue_rows:
                line.set_color("crimson")
                line.set_linewidth(2.2)
                line.set_zorder(8)
                continue
            base = self.segment_base_colors[i] if i < len(self.segment_base_colors) else "gray"
            line.set_color(base)
            line.set_linewidth(1.5)
            line.set_zorder(3)
        self._apply_material_isolation()

    def _clear_normals(self):
        for art in self.normal_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.normal_artists = []

    def _segment_primitives(self, seg: 'Segment') -> 'List[Tuple[float, float, float, float]]':
        count = min(len(seg.x), len(seg.y))
        n_pairs = count // 2
        out: 'List[Tuple[float, float, float, float]]' = []
        for i in range(n_pairs):
            idx = 2 * i
            out.append((seg.x[idx], seg.y[idx], seg.x[idx + 1], seg.y[idx + 1]))
        return out

    def _segment_plot_xy(self, seg: 'Segment') -> 'Tuple[List[float], List[float]]':
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)

        xs: 'List[float]' = []
        ys: 'List[float]' = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            if i == 0 or (xs[-1], ys[-1]) != (x1, y1):
                if i:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                xs.append(x1)
                ys.append(y1)
            xs.append(x2)
            ys.append(y2)

        if not xs or not ys:
            return list(seg.x), list(seg.y)
        return xs, ys


    _AIR_COLOR = "#4da6ff"
    _PEC_COLOR = "#6e6e6e"
    _SHEET_COLOR = "#d4a017"
    _DIEL_COLORS = ["#2e9e4f", "#7f5bd4", "#0fa3a3", "#c46a1b", "#b33f8e", "#8a9a1a"]

    def _diel_color(self, flag: 'int') -> 'str':
        return self._DIEL_COLORS[(max(int(flag), 1) - 1) % len(self._DIEL_COLORS)]

    def _segment_side_materials(self, seg: 'Segment') -> 'Tuple[str, str, str, str]':
        """Return (front_label, front_color, back_label, back_color)."""

        props = self._ensure_prop_len(seg.properties, 5)
        seg_type = self._parse_int_token(props[0], 2)
        pos_mat = self._parse_int_token(props[3], 0)
        neg_mat = self._parse_int_token(props[4], 0)
        if seg_type == 1:
            return "air", self._SHEET_COLOR, "air", self._SHEET_COLOR
        if seg_type == 2:
            if self.geometry_mode() == "bor":
                kinds = {self._parse_int_token(item.properties[0], -1)
                         if item.properties else -1 for item in self.segments}
                if kinds == {1, 2}:
                    # The BoR solver treats pure PEC portions of a mixed
                    # transmitting sheet as zero-Z sheet, with air on both sides.
                    return "air", self._AIR_COLOR, "air", self._AIR_COLOR
            return "air", self._AIR_COLOR, "PEC", self._PEC_COLOR
        if seg_type == 3:
            return "air", self._AIR_COLOR, f"d{pos_mat}", self._diel_color(pos_mat)
        if seg_type == 4:
            return f"d{pos_mat}", self._diel_color(pos_mat), "PEC", self._PEC_COLOR
        if seg_type == 5:
            return f"d{pos_mat}", self._diel_color(pos_mat), f"d{neg_mat}", self._diel_color(neg_mat)
        return "?", "magenta", "?", "magenta"

    def _render_normals(self, ax=None):
        main = ax is None
        if main:
            self._clear_normals()
            self._normal_view = self._preview_view_key()
            self._detail_dirty = True
        artists = self.normal_artists if main else []
        if not self.chk_show_normals.isChecked() or not self.segments:
            return artists
        from ghost_backend.geometry.preview import select_normal_samples, visible_segment_midpoint
        ax = self.canvas.ax if main else ax
        ax.apply_aspect()
        if main:
            self._normal_view = self._preview_view_key()
        transform, inverse = ax.transData, ax.transData.inverted()
        selected = {i.row() for i in self.table.selectedIndexes()}
        if self._selected_row is not None:
            selected.add(self._selected_row)
        scope = self.cmb_normal_scope.currentData()
        candidates = []
        bounds = (ax.bbox.x0, ax.bbox.y0, ax.bbox.x1, ax.bbox.y1)
        for row, seg in enumerate(self.segments):
            if not self._inspection_row_matches(row):
                continue
            if scope == "selected" and row not in selected:
                continue
            front_label, front_color, back_label, back_color = self._segment_side_materials(seg)
            for x1, y1, x2, y2 in self._segment_primitives(seg):
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if not math.isfinite(length) or length <= 1e-12:
                    continue
                start_px, end_px = transform.transform([(x1, y1), (x2, y2)])
                clipped = visible_segment_midpoint(start_px, end_px, bounds)
                if clipped is None:
                    continue
                if all(bounds[0] <= p[0] <= bounds[2] and bounds[1] <= p[1] <= bounds[3]
                       for p in (start_px, end_px)):
                    mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                else:
                    mx, my = inverse.transform(clipped)
                center = transform.transform((mx, my))
                nx, ny = -dy / length, dx / length
                # Transform a direction without translation: subtracting two
                # transformed positions introduces noise in zero components.
                matrix = transform.get_affine().get_matrix()
                screen_length = math.hypot(matrix[0, 0] * nx + matrix[0, 1] * ny,
                                           matrix[1, 0] * nx + matrix[1, 1] * ny)
                if screen_length <= 0:
                    continue
                arrow_dx, arrow_dy = nx * 18 / screen_length, ny * 18 / screen_length
                back = (mx - nx * 7 / screen_length, my - ny * 7 / screen_length)
                candidates.append(dict(row=row, midpoint=tuple(center),
                    priority=0 if row in selected else (1 if row in self.issue_rows else 2),
                    arrow=(mx, my, arrow_dx, arrow_dy), tick=((mx, my), tuple(back)),
                    front="crimson" if row in self.issue_rows else front_color, back=back_color))
        samples = candidates if scope == "all" else select_normal_samples(candidates, min_spacing=38, bounds=bounds)
        if samples:
            x, y, u, v = zip(*(sample["arrow"] for sample in samples))
            artists.append(ax.quiver(x, y, u, v, angles="xy", scale_units="xy",
                scale=1, color=[s["front"] for s in samples], units="dots", width=1.3,
                headwidth=4, headlength=5, alpha=.95, zorder=12))
            collection = LineCollection([s["tick"] for s in samples],
                colors=[s["back"] for s in samples], linewidths=1.6, alpha=.9, zorder=12)
            ax.add_collection(collection, autolim=False)
            artists.append(collection)
        if main and self._selected_row is not None and 0 <= self._selected_row < len(self.segments):
            seg = self.segments[self._selected_row]
            front, front_color, back, _ = self._segment_side_materials(seg)
            artists.append(ax.text(.01, .99,
                f"{seg.name}: arrow into {front} | behind {back}", transform=ax.transAxes,
                va="top", fontsize=8, color="#222222",
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": front_color, "alpha": .95},
                zorder=15))
        return artists

    def _on_show_normals_toggled(self, checked: 'bool'):
        _ = checked
        self._render_normals()
        self.canvas.draw_idle()


    def _clear_fills(self):
        for art in self.fill_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.fill_artists = []

    def _fill_loops(self) -> 'List[Dict[str, Any]]':
        from ghost_backend.geometry.preview import build_material_faces
        return build_material_faces(self.segments, self._segment_side_materials,
                                    close_axis=self.geometry_mode() == "bor")

    def _render_fills(self):
        self._clear_fills()
        self._detail_dirty = True
        self._fill_loops_cache = []
        if not self.chk_fill_materials.isChecked() or not self.segments:
            return
        self._fill_loops_cache = self._fill_loops()
        self.fill_artists = self._draw_fill_faces(self.canvas.ax, legend=True)

    def _on_fill_materials_toggled(self, checked: 'bool'):
        _ = checked
        self._render_fills()
        self.canvas.draw_idle()


    def _ibcs_lookup(self) -> 'Dict[int, Dict[str, Any]]':
        """Build a {flag: info} map from the live IBCS table.

        info keys: 'kind' ('linear'/'cosine'/'exp'/'tabulated'/'undefined'),
                   'z_start' (complex), 'z_end' (complex), 'raw' (token row).
        """
        rows = self._read_small_table(self.table_ibc)
        out: 'Dict[int, Dict[str, Any]]' = {}
        for row in rows:
            if not row:
                continue
            flag = self._parse_int_token(row[0], 0)
            if flag <= 0:
                continue
            if len(row) >= 2 and row[1].lower() == "thin_dielectric":
                out[flag] = {"kind": "thin_dielectric", "z_start": None, "z_end": None, "raw": row}
                continue
            if is_tabulated_row(row):
                out[flag] = {
                    "kind": "tabulated",
                    "filename": material_filename_from_row(row),
                    "z_start": None,
                    "z_end": None,
                    "raw": row,
                }
                continue
            if is_ibc_inline_row(row):
                kind = str(row[1]).strip().lower()
                r_s = self._parse_float_token(row[2], 0.0)
                x_s = self._parse_float_token(row[3], 0.0)
                if kind == "constant":
                    z_s = complex(r_s, x_s)
                    z_e = z_s
                else:
                    r_e = self._parse_float_token(row[4], 0.0)
                    x_e = self._parse_float_token(row[5], 0.0)
                    z_s = complex(r_s, x_s)
                    z_e = complex(r_e, x_e)
                out[flag] = {
                    "kind": kind,
                    "z_start": z_s,
                    "z_end": z_e,
                    "raw": row,
                }
                continue
            out[flag] = {"kind": "undefined", "z_start": None, "z_end": None, "raw": row}
        return out

    def _format_z(self, z: 'Optional[complex]') -> 'str':
        if z is None:
            return "?"
        return f"{z.real:g}{'+' if z.imag >= 0 else '-'}{abs(z.imag):g}j ohm"

    def _resolve_segment_bc(self, seg: 'Segment', lookup: 'Optional[Dict[int, Dict[str, Any]]]' = None) -> 'str':
        """Human-readable resolved boundary condition for a segment."""
        props = list(seg.properties)
        seg_type = self._parse_int_token(props[0] if props else "", -1)
        ibc = self._parse_int_token(props[2] if len(props) >= 3 else "", 0)
        pos_mat = self._parse_int_token(props[3] if len(props) >= 4 else "", 0)
        neg_mat = self._parse_int_token(props[4] if len(props) >= 5 else "", 0)


        if seg_type == 1:
            base = "TYPE 1 . free-floating sheet"
        elif seg_type == 2:
            base = "TYPE 2 . PEC" if ibc == 0 else f"TYPE 2 . IBC-coated PEC"
        elif seg_type in (3, 4, 5):
            dstr = f"pos_mat={pos_mat}" + (f", neg_mat={neg_mat}" if seg_type == 5 else "")
            base = f"TYPE {seg_type} . dielectric interface ({dstr})"
        else:
            base = f"TYPE {seg_type}"

        if ibc == 0:
            return base
        lut = lookup if lookup is not None else self._ibcs_lookup()
        info = lut.get(ibc)
        if info is None:
            return f"{base}  |  IBC {ibc} (NOT DEFINED)"
        kind = info["kind"]
        if kind == "thin_dielectric":
            row = info["raw"]
            try:
                thickness_in = float(row[2]) / METERS_PER_INCH
            except (ValueError, IndexError):
                return f"{base} | malformed thin layer"
            return f"{base}  |  thin layer {thickness_in:.6g} in, dielectric {row[3]}" if len(row) == 4 else f"{base} | malformed thin layer"
        if kind == "tabulated":
            return (
                f"{base}  |  IBC {ibc} -> tabulated "
                f"({info.get('filename', '?')})"
            )
        if kind == "undefined":
            return f"{base}  |  IBC {ibc} (malformed row)"
        z1, z2 = info["z_start"], info["z_end"]
        if z1 == z2:
            return f"{base}  |  IBC {ibc} -> constant {self._format_z(z1)}"
        return (
            f"{base}  |  IBC {ibc} -> taper({kind})  "
            f"start {self._format_z(z1)}  ->  end {self._format_z(z2)}"
        )

    def _z_to_color(self, z: 'Optional[complex]', z_ref_mag: 'float') -> 'Tuple[float, float, float]':
        """Map an impedance value to an RGB colour.

        * |Z| near zero  -> near-black (PEC-like)
        * |Z| near 377 ohm -> mid-blue (free-space-like)
        * |Z| large      -> light blue / washed out
        Reactance tints warm (|X| large -> toward magenta).
        """
        if z is None:
            return (0.55, 0.55, 0.55)
        mag = abs(z)

        t = min(1.0, mag / max(1.0, 2.0 * 377.0))

        r = 0.05 + 0.75 * t
        g = 0.10 + 0.55 * t
        b = 0.25 + 0.70 * (1.0 - abs(t - 0.5) * 2.0)

        if mag > 1e-12:
            react_frac = min(1.0, abs(z.imag) / mag)
            r = min(1.0, r + 0.25 * react_frac)
        return (r, g, b)

    def _clear_impedance_overlay(self):
        for artist in self.impedance_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.impedance_artists = []

    def _render_impedance_overlay(self):
        self._clear_impedance_overlay()
        self._detail_dirty = True
        if not self.chk_show_impedance.isChecked() or not self.segments:
            return
        ax = self.canvas.ax
        all_x = [x for seg in self.segments for x in seg.x]
        all_y = [y for seg in self.segments for y in seg.y]
        if not all_x or not all_y:
            return
        diag = max(((max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2) ** 0.5, 1.0)
        marker_size = max(4.0, 0.015 * diag * 100)

        lookup = self._ibcs_lookup()

        for row, seg in enumerate(self.segments):
            if not self._inspection_row_matches(row):
                continue
            primitives = self._segment_primitives(seg)
            if not primitives:
                continue
            props = list(seg.properties)
            seg_type = self._parse_int_token(props[0] if props else "", -1)
            ibc = self._parse_int_token(props[2] if len(props) >= 3 else "", 0)


            if seg_type in (1, 2, 3, 4, 5) and ibc != 0 and ibc in lookup:
                info = lookup[ibc]
                if info["kind"] == "tabulated":
                    c_start = c_end = (0.95, 0.55, 0.10)
                else:
                    c_start = self._z_to_color(info["z_start"], 377.0)
                    c_end = self._z_to_color(info["z_end"], 377.0)
            elif seg_type in (1, 2, 3, 4, 5) and ibc == 0:

                c_start = c_end = (0.05, 0.05, 0.08)
            else:
                c_start = c_end = (0.55, 0.55, 0.55)


            seg_lens = []
            for x1, y1, x2, y2 in primitives:
                seg_lens.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            total_len = sum(seg_lens) or 1.0
            cum = 0.0
            SAMPLES_PER_PRIM = 12
            for (x1, y1, x2, y2), L in zip(primitives, seg_lens):
                s0 = cum / total_len
                s1 = (cum + L) / total_len
                cum += L


                for k in range(SAMPLES_PER_PRIM):
                    u0 = k / SAMPLES_PER_PRIM
                    u1 = (k + 1) / SAMPLES_PER_PRIM
                    px0 = x1 + u0 * (x2 - x1); py0 = y1 + u0 * (y2 - y1)
                    px1 = x1 + u1 * (x2 - x1); py1 = y1 + u1 * (y2 - y1)

                    s_mid = s0 + 0.5 * (u0 + u1) * (s1 - s0)
                    cr = c_start[0] + s_mid * (c_end[0] - c_start[0])
                    cg = c_start[1] + s_mid * (c_end[1] - c_start[1])
                    cb = c_start[2] + s_mid * (c_end[2] - c_start[2])
                    line = ax.plot(
                        [px0, px1], [py0, py1],
                        color=(cr, cg, cb), lw=4.0, solid_capstyle="butt",
                        alpha=0.7, zorder=8,
                    )
                    self.impedance_artists.extend(line)


            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            m_start = ax.scatter([sx], [sy], s=marker_size, marker="o",
                                  facecolor="#1f9e3c", edgecolor="black", lw=0.6, zorder=13)
            m_end = ax.scatter([ex], [ey], s=marker_size, marker="o",
                                facecolor="#d93023", edgecolor="black", lw=0.6, zorder=13)
            self.impedance_artists.append(m_start)
            self.impedance_artists.append(m_end)

    def _on_show_impedance_toggled(self, checked: 'bool'):
        _ = checked
        self._render_impedance_overlay()
        self.canvas.draw_idle()

    def _update_status_label(self, row: 'int'):
        if row < 0 or row >= len(self.segments):
            self.lbl_status.setText("")
            return
        seg = self.segments[row]
        self.lbl_status.setText(f"{seg.name}  |  {self._resolve_segment_bc(seg)}")

    def _parse_int_token(self, token: 'str', default: 'int' = 0) -> 'int':
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("mat."):
            text = text.split("mat.", 1)[1]
        try:
            return int(float(text))
        except ValueError:
            return default

    def _parse_float_token(self, token: 'str', default: 'float' = 0.0) -> 'float':
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _point_key(self, x: 'float', y: 'float', tol: 'float') -> 'Tuple[int, int]':
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))

    def _segments_intersect(
        self,
        a1: 'Tuple[float, float]',
        a2: 'Tuple[float, float]',
        b1: 'Tuple[float, float]',
        b2: 'Tuple[float, float]',
        tol: 'float',
    ) -> 'bool':
        ax1, ay1 = a1
        ax2, ay2 = a2
        bx1, by1 = b1
        bx2, by2 = b2

        min_ax, max_ax = min(ax1, ax2), max(ax1, ax2)
        min_ay, max_ay = min(ay1, ay2), max(ay1, ay2)
        min_bx, max_bx = min(bx1, bx2), max(bx1, bx2)
        min_by, max_by = min(by1, by2), max(by1, by2)
        if max_ax < min_bx - tol or max_bx < min_ax - tol:
            return False
        if max_ay < min_by - tol or max_by < min_ay - tol:
            return False

        def orient(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'float':
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        def on_seg(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'bool':
            return (
                min(px, qx) - tol <= rx <= max(px, qx) + tol
                and min(py, qy) - tol <= ry <= max(py, qy) + tol
            )

        o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

        if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (
            o3 > tol and o4 < -tol or o3 < -tol and o4 > tol
        ):
            return True

        if abs(o1) <= tol and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if abs(o2) <= tol and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if abs(o3) <= tol and on_seg(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if abs(o4) <= tol and on_seg(bx1, by1, bx2, by2, ax2, ay2):
            return True
        return False

    def _cancel_validation(self, *_):
        self._validation_version += 1
        self.issue_rows.clear()
        self.validation_results.hide()
        self.validation_results.setRowCount(0)
        self._refresh_segment_styles()
        if self._validation_worker is not None:
            self._validation_worker.abort.set()

    def validate_geometry(self):
        if self._validation_worker is not None:
            self._cancel_validation()
            self.lbl_status.setText('Canceling validation...')
            return
        from ghost_backend.geometry.validation import GeometryAudit
        material_dir = os.path.dirname(os.path.abspath(self.loaded_path)) if self.loaded_path else os.getcwd()
        worker = _GeometryValidationWorker()
        worker.audit = GeometryAudit(copy.deepcopy(self.segments),
            self._read_small_table(self.table_ibc), self._read_small_table(self.table_diel),
            material_dir, worker.checkpoint, mode=self.geometry_mode())
        worker.version = self._validation_version
        self._validation_worker = worker
        worker.ready.connect(self._validation_ready)
        worker.failed.connect(self._validation_failed)
        worker.finished.connect(self._validation_finished)
        worker.finished.connect(worker.deleteLater)
        self.destroyed.connect(worker.abort.set)
        self.btn_validate.setText('Cancel validation')
        self.lbl_status.setText(f'Validating captured {self.geometry_mode().upper()} geometry...')
        worker.start()

    def _validation_finished(self):
        self._validation_worker = None
        self.btn_validate.setText('Validate')

    def _validation_failed(self, message):
        self.lbl_status.setText(message)

    def _validation_ready(self, version, result):
        if version != self._validation_version:
            self.lbl_status.setText('Geometry changed during validation. Validate again for current results.')
            return
        findings, issue_rows = result
        ordered = sorted(findings, key=lambda f: {"ERROR": 0, "WARN": 1, "INFO": 2}.get(f[0], 3))
        self.validation_results.setRowCount(len(ordered))
        for index, (level, source_row, message) in enumerate(ordered):
            for column, value in enumerate((level, str(source_row + 1) if source_row >= 0 else "-", message)):
                item = QTableWidgetItem(value)
                item.setToolTip(message)
                item.setData(Qt.UserRole, source_row)
                self.validation_results.setItem(index, column, item)
        self.validation_results.resizeRowsToContents()
        self.validation_results.setVisible(bool(ordered))
        self.issue_rows = issue_rows
        self._refresh_segment_styles()
        self._render_normals()
        self._render_fills()
        self.canvas.draw_idle()

        errors = [msg for level, _, msg in findings if level == "ERROR"]
        warns = [msg for level, _, msg in findings if level == "WARN"]
        infos = [msg for level, _, msg in findings if level == "INFO"]

        summary = (
            f"Validation complete: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info message(s)."
        )
        self.lbl_status.setText(summary + " Click a finding to locate it.")
        detail_lines = errors + warns + infos
        if detail_lines:
            max_lines = 30
            shown = detail_lines[:max_lines]
            detail_text = "\n".join(shown)
            if len(detail_lines) > max_lines:
                detail_text += f"\n... ({len(detail_lines) - max_lines} additional message(s))"
            message = summary + "\n\n" + detail_text
        else:
            message = summary + "\nNo issues found."

        if errors or warns:
            QMessageBox.warning(self, "Geometry Validation", message)
        else:
            QMessageBox.information(self, "Geometry Validation", message)

    def save_geo(self):
        default_name = f"geometry_out{self._last_ext}"
        fname, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Geometry File", default_name, "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return False
        fname = self._ensure_extension(fname, selected_filter)
        self._last_ext = os.path.splitext(fname)[1].lower()
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        try:
            text = build_geometry_text(self.title, self.segments, ibcs_rows, dielectric_rows)
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
            return False

        target = Path(fname).expanduser().resolve(strict=False)
        source_geometry = (
            Path(self.loaded_path).expanduser().resolve(strict=False)
            if self.loaded_path
            else None
        )
        source_folder = source_geometry.parent if source_geometry else None


        material_names: 'Dict[str, str]' = {}
        try:
            for row in list(ibcs_rows) + list(dielectric_rows):
                name = material_filename_from_row(row)
                if name is None:
                    continue
                key = name.casefold()
                previous_name = material_names.get(key)
                if previous_name is not None and previous_name != name:
                    raise ValueError(
                        "Material filenames that differ only by letter case "
                        f"are not portable: {previous_name!r} and {name!r}."
                    )
                material_names[key] = name

            target_folder_key = os.path.normcase(str(target.parent))
            source_folder_key = (
                os.path.normcase(str(source_folder))
                if source_folder is not None
                else None
            )
            sidecar_copies: 'List[Tuple[Path, Path]]' = []
            for name in sorted(material_names.values(), key=str.casefold):
                destination = target.parent / name
                if source_folder_key == target_folder_key:
                    if not destination.is_file():
                        raise FileNotFoundError(
                            f"Referenced material sidecar is missing beside "
                            f"the geometry: {destination}"
                        )
                    continue
                if source_folder is None:


                    if not destination.is_file():
                        raise FileNotFoundError(
                            f"Referenced material sidecar is not in the save "
                            f"directory: {destination}"
                        )
                    continue

                source = source_folder / name
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Referenced material sidecar is missing beside the "
                        f"current geometry: {source}"
                    )
                if destination.exists() and not destination.is_file():
                    raise OSError(
                        f"Material sidecar target is not a regular file: "
                        f"{destination}"
                    )
                sidecar_copies.append((source, destination))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Geometry Save Blocked",
                f"Could not preserve the geometry's material sidecars:\n{e}",
            )
            return False

        replacements = [
            destination
            for _source, destination in sidecar_copies
            if destination.exists()
        ]
        if replacements:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            shown = "\n".join(f"  - {path.name}" for path in replacements[:12])
            if len(replacements) > 12:
                shown += f"\n  - ... and {len(replacements) - 12} more"
            answer = QMessageBox.question(
                self,
                "Replace Material Sidecars?",
                "Saving this geometry in the selected directory also needs "
                "to replace these material files:\n\n"
                f"{shown}\n\n"
                "Replace them with the versions beside the current geometry?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                return False

        transaction = AtomicFileTransaction()
        try:


            for source, destination in sidecar_copies:
                transaction.stage_copy(source, destination)
            transaction.stage_text(text, target)
            transaction.publish()
        except Exception as e:
            try:
                transaction.abort()
            except Exception as rollback_error:
                QMessageBox.critical(
                    self,
                    "Geometry Save Failed",
                    f"Failed to save file: {e}\n\n"
                    f"Rollback also reported: {rollback_error}",
                )
                return False
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            return False
        transaction.commit()
        self.loaded_path = str(target)
        self._set_dirty(False)
        QMessageBox.information(self, "Saved", f"Geometry saved to {target}")
        return True

    def _read_small_table(self, table: 'QTableWidget') -> 'List[List[str]]':
        rows: 'List[List[str]]' = []
        for r in range(table.rowCount()):
            tokens: 'List[str]' = []
            for c in range(table.columnCount()):
                widget = table.cellWidget(r, c)
                if isinstance(widget, QComboBox):
                    data = widget.currentData()
                    val = str(data) if data is not None else widget.currentText().strip()
                else:
                    item = table.item(r, c)
                    val = item.text().strip() if item else ""
                tokens.append(val)
            while tokens and tokens[-1] == "":
                tokens.pop()
            if table is self.table_ibc and len(tokens) >= 3 and tokens[1].lower() == "thin_dielectric":
                item = table.item(r, 2)
                saved = item.data(Qt.UserRole) if item is not None else None
                if saved is not None and tokens[2] == saved[0]:
                    tokens[2] = saved[1]
                else:
                    try:
                        tokens[2] = format(float(tokens[2]) * METERS_PER_INCH, ".17g")
                    except (ValueError, OverflowError):
                        pass
            if tokens:
                rows.append(tokens)
        return rows

    def _set_equal_column_widths(self, table: 'QTableWidget', enabled: 'bool' = True):
        header = table.horizontalHeader()
        if not header:
            return
        if enabled:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.Interactive)

    def _ensure_extension(self, fname: 'str', selected_filter: 'str') -> 'str':
        root, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in (".geo", ".txt"):
            return fname
        filt = (selected_filter or "").lower()
        if ".geo" in filt:
            return root + ".geo"
        if ".txt" in filt:
            return root + ".txt"
        return root + ".geo"

    def get_geometry_snapshot(self) -> 'Dict[str, Any]':
        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        snapshot = build_geometry_snapshot(
            self.title,
            self.segments,
            ibcs_rows,
            dielectric_rows,
        )
        snapshot["source_path"] = self.loaded_path
        return snapshot
