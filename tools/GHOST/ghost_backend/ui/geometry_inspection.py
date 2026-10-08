"""Geometry inspection controls; all state here is view-only."""
import math
import copy
import threading

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton
except ImportError:
    from PySide2.QtCore import Qt, QThread, Signal
    from PySide2.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch, Rectangle, ConnectionPatch
from ghost_backend.ui.table_editors import ScrollSafeComboBox
from ghost_backend.geometry.measurements import point_to_primitive, closest_segment_points



class _GapWorker(QThread):
    ready = Signal(int, object)
    failed = Signal(int, str)
    active = set()

    def __init__(self, version, rows, segments):
        super().__init__()
        self.version, self.rows, self.segments = version, rows, segments
        self.abort = threading.Event()
        self.active.add(self)
        self.finished.connect(lambda: self.active.discard(self))

    def checkpoint(self):
        if self.abort.is_set():
            raise InterruptedError("Minimum gap canceled.")

    def run(self):
        try:
            result = closest_segment_points(*self.segments, checkpoint=self.checkpoint)
            self.checkpoint()
            self.ready.emit(self.version, (self.rows, result))
        except Exception as exc:
            self.failed.emit(self.version, str(exc))


class GeometryInspectionMixin:
    def _init_inspection(self, plot_layout):
        self._gap_anchor = None
        self._gap_version = 0
        self._gap_worker = None
        self._gap_result = None
        self._gap_artists = []
        self._detail_ax = None
        self._detail_center = None
        self._detail_sources = []
        self._detail_dirty = True
        self._detail_view = None
        self._fill_loops_cache = []

        isolation = QHBoxLayout()
        isolation.addWidget(QLabel("Material"))
        self.cmb_material_isolation = ScrollSafeComboBox()
        self.cmb_material_isolation.addItem("All materials", None)
        self.cmb_material_isolation.setToolTip("Show boundaries touching the selected material and emphasize its regions.")
        isolation.addWidget(self.cmb_material_isolation)
        self.chk_isolation_context = QCheckBox("Faint context")
        self.chk_isolation_context.setChecked(True)
        self.chk_isolation_context.setToolTip("Retain other materials faintly for orientation. Their boundaries cannot be picked while isolated.")
        isolation.addWidget(self.chk_isolation_context)
        self.chk_detail_inset = QCheckBox("Detail inset")
        self.chk_detail_inset.setToolTip("Click the overview to position the magnified detail. Measure directly inside it; scroll inside to change magnification.")
        isolation.addWidget(self.chk_detail_inset)
        self.cmb_detail_zoom = ScrollSafeComboBox()
        for zoom in (8, 20, 50):
            self.cmb_detail_zoom.addItem(f"{zoom}x", zoom)
        self.cmb_detail_zoom.setCurrentIndex(1)
        self.cmb_detail_zoom.setToolTip("Screen magnification relative to the overview.")
        isolation.addWidget(self.cmb_detail_zoom)
        isolation.addStretch(1)
        plot_layout.insertLayout(3, isolation)

        ruler = QHBoxLayout()
        self.btn_measure_gap = QPushButton("Measure gap")
        self.btn_measure_gap.setCheckable(True)
        self.btn_measure_gap.setToolTip("Click a point on the first boundary, then another primitive. Measure from the snapped first point to the nearest point on the second primitive.")
        self.btn_min_gap = QPushButton("Min selected gap")
        self.btn_min_gap.setEnabled(False)
        self.btn_min_gap.setToolTip("Select exactly two rows. Shared endpoints and crossings count as a zero minimum; use Measure gap for local thickness away from a junction.")
        self.btn_clear_gap = QPushButton("Clear ruler")
        self.cmb_geometry_units = ScrollSafeComboBox()
        self.cmb_geometry_units.addItems(["inches", "meters"])
        self.cmb_geometry_units.setToolTip("Interpret the stored coordinates in these units, synchronized with Solver units. Does not rescale the drawing.")
        for widget in (self.btn_measure_gap, self.btn_min_gap, self.btn_clear_gap,
                       QLabel("Units"), self.cmb_geometry_units):
            ruler.addWidget(widget)
        ruler.addStretch(1)
        plot_layout.insertLayout(4, ruler)
        self.lbl_gap = QLabel("Ruler: click two boundaries, or select two rows for their minimum gap.")
        self.lbl_gap.setWordWrap(True)
        self.lbl_gap.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_gap.setMinimumHeight(self.lbl_gap.fontMetrics().height() * 2)
        plot_layout.insertWidget(5, self.lbl_gap)

        self.cmb_material_isolation.currentIndexChanged.connect(self._on_isolation_changed)
        self.chk_isolation_context.toggled.connect(self._on_isolation_changed)
        self.chk_detail_inset.toggled.connect(self._on_detail_changed)
        self.cmb_detail_zoom.currentIndexChanged.connect(self._on_detail_changed)
        self.btn_measure_gap.toggled.connect(self._on_measure_gap_toggled)
        self.btn_min_gap.clicked.connect(self._measure_selected_gap)
        self.btn_clear_gap.clicked.connect(self._clear_gap)
        self.cmb_geometry_units.currentTextChanged.connect(self._inspection_units_changed)
        self.geometry_changed.connect(self._inspection_geometry_changed)
        self.cmb_geometry_mode.currentIndexChanged.connect(self._inspection_geometry_changed)
        self.canvas.mpl_connect("key_press_event", self._inspection_key_press)

    def geometry_units(self):
        return self.cmb_geometry_units.currentText()

    def set_geometry_units(self, units):
        if str(units).lower() not in ("inches", "meters"):
            raise ValueError("Geometry units must be inches or meters.")
        self.cmb_geometry_units.setCurrentText(str(units).lower())

    def _geometry_unit_label(self):
        return "in" if self.geometry_units() == "inches" else "m"

    def _inspection_units_changed(self, *_):
        self._update_geometry_axes()
        self._update_gap_label()
        self._render_gap()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _refresh_material_choices(self):
        labels = set()
        for segment in self.segments:
            front, _, back, _ = self._segment_side_materials(segment)
            labels.update((front, back))
        previous = self.cmb_material_isolation.currentData()
        self.cmb_material_isolation.blockSignals(True)
        try:
            self.cmb_material_isolation.clear()
            self.cmb_material_isolation.addItem("All materials", None)
            for label in sorted(labels, key=lambda label: (label not in ("air", "PEC"), label)):
                self.cmb_material_isolation.addItem(label, label)
            index = self.cmb_material_isolation.findData(previous)
            self.cmb_material_isolation.setCurrentIndex(max(index, 0))
        finally:
            self.cmb_material_isolation.blockSignals(False)

    def _inspection_row_matches(self, row):
        if not hasattr(self, "cmb_material_isolation"):
            return True
        target = self.cmb_material_isolation.currentData()
        if target is None:
            return True
        front, _, back, _ = self._segment_side_materials(self.segments[row])
        return target in (front, back)

    def _apply_material_isolation(self):
        if not hasattr(self, "cmb_material_isolation"):
            return
        context = self.chk_isolation_context.isChecked()
        for row, line in enumerate(self.segment_lines):
            matches = self._inspection_row_matches(row)
            line.set_visible(matches or context)
            line.set_alpha(1.0 if matches else .12)
        self._detail_dirty = True

    def _on_isolation_changed(self, *_):
        referenced_rows = set()
        if self._gap_anchor is not None:
            referenced_rows.add(self._gap_anchor["row"])
        if self._gap_result is not None:
            referenced_rows.update((self._gap_result["first_row"], self._gap_result["second_row"]))
        if self._gap_worker is not None:
            referenced_rows.update(self._gap_worker.rows)
        if any(not self._inspection_row_matches(row) for row in referenced_rows):
            self._clear_gap()
        self._refresh_segment_styles()
        self._render_normals()
        self._render_impedance_overlay()
        self._render_fills()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _material_face_patch(self, face, ax):
        target = self.cmb_material_isolation.currentData()
        matches = target is None or (face["consistent"] and face["label"] == target)
        background = to_rgb(ax.get_facecolor())
        if not matches:
            # Keep every mask opaque. Removing a nested nonmatching region
            # would incorrectly fill the hole with its parent's material.
            color = tuple(.93 * value + .07 * .5 for value in background) if self.chk_isolation_context.isChecked() else background
            edge, hatch = "none", None
        elif not face["consistent"]:
            color, edge, hatch = background, "red", "//"
        elif face["label"] == "air":
            color, edge, hatch = background, "none", None
        else:
            color = tuple(.55 * value + .45 for value in to_rgb(face["color"]))
            edge, hatch = "none", None
        return Polygon(face["points"], closed=True, facecolor=color,
                       edgecolor=edge, hatch=hatch, linewidth=0, alpha=1,
                       zorder=1 + .01 * face["depth"])

    def _draw_fill_faces(self, ax, *, legend=False):
        artists, labels = [], {}
        target = self.cmb_material_isolation.currentData()
        for face in sorted(self._fill_loops_cache, key=lambda face: face["depth"]):
            patch = self._material_face_patch(face, ax)
            ax.add_patch(patch)
            artists.append(patch)
            if face["consistent"] and face["label"] != "air" and target in (None, face["label"]):
                labels[face["label"]] = face["color"]
        if legend and labels:
            key = ax.legend(handles=[Patch(facecolor=color, label=label) for label, color in sorted(labels.items())],
                            loc="lower left", fontsize=7, framealpha=.9)
            key.set_zorder(14)
            artists.append(key)
        return artists

    def _inspection_geometry_changed(self, *_):
        self._clear_gap()
        previous = self.cmb_material_isolation.currentData()
        self._refresh_material_choices()
        self._apply_material_isolation()
        if previous is not None:
            self._render_normals()
            self._render_impedance_overlay()
            self._render_fills()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _inspection_selection_changed(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        self.btn_min_gap.setEnabled(len(rows) == 2 or self._gap_worker is not None)
        if self._selected_row is not None and self.chk_detail_inset.isChecked():
            primitives = self._segment_primitives(self.segments[self._selected_row])
            if primitives:
                x1, y1, x2, y2 = primitives[len(primitives) // 2]
                self._detail_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self._detail_dirty = True

    def _inspection_axes(self, event):
        axes = getattr(event, "inaxes", None)
        if axes is self.canvas.ax or (self._detail_ax is not None and axes is self._detail_ax):
            return axes
        return None

    def _nearest_primitive(self, event):
        axes = self._inspection_axes(event)
        if event.x is None or event.y is None or axes is None:
            return None
        best, best_distance = None, 7.0
        for row, segment in enumerate(self.segments):
            if not self._inspection_row_matches(row):
                continue
            for index, primitive in enumerate(self._segment_primitives(segment)):
                if not all(math.isfinite(value) for value in primitive):
                    continue
                start, end = axes.transData.transform([primitive[:2], primitive[2:]])
                point, distance = point_to_primitive((event.x, event.y), (*start, *end))
                if distance < best_distance:
                    best_distance = distance
                    best = dict(row=row, primitive=index,
                                point=tuple(axes.transData.inverted().transform(point)),
                                coordinates=primitive)
        return best

    def _on_measure_gap_toggled(self, checked):
        if checked and self._gap_worker is not None:
            self._clear_gap()
        if checked and self.toolbar.mode:
            if "pan" in str(self.toolbar.mode).lower():
                self.toolbar.pan()
            else:
                self.toolbar.zoom()
        self._gap_anchor = None
        if checked:
            self.lbl_gap.setText("Local gap: click the first boundary, then the second. Use the inset to pick closely spaced edges.")
        else:
            self._update_gap_label()
        self._render_gap()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _inspection_key_press(self, event):
        if getattr(event, "key", None) == "escape":
            self.btn_measure_gap.setChecked(False)

    def _inspection_button_press(self, event):
        if self.toolbar.mode or event.inaxes not in (self.canvas.ax, self._detail_ax) or event.inaxes is None:
            return False
        if event.button != 1:
            return False
        if self.btn_measure_gap.isChecked():
            hit = self._nearest_primitive(event)
            if hit is None:
                self.lbl_gap.setText("Click closer to a visible boundary. The ruler snaps to actual primitives.")
                return True
            if self._gap_anchor is None:
                self._gap_anchor = hit
                self._gap_result = None
                self.lbl_gap.setText(f"First boundary: {self.segments[hit['row']].name}. Now click the second boundary.")
                self._detail_center = hit["point"]
            else:
                first = self._gap_anchor
                if (first["row"], first["primitive"]) == (hit["row"], hit["primitive"]):
                    self.lbl_gap.setText("Choose a different primitive for the second boundary.")
                    return True
                second, distance = point_to_primitive(first["point"], hit["coordinates"])
                self._gap_result = dict(first_point=first["point"], second_point=second, distance=distance,
                                        first_row=first["row"], second_row=hit["row"],
                                        first_primitive=first["primitive"], second_primitive=hit["primitive"],
                                        kind="Local gap")
                self._gap_anchor = None
                self._detail_center = tuple((a + b) / 2 for a, b in zip(first["point"], second))
                self._update_gap_label()
            self._render_gap()
            self._detail_dirty = True
            self.canvas.draw_idle()
            return True
        if event.inaxes is self._detail_ax:
            hit = self._nearest_primitive(event)
            if hit is not None:
                # Keep the current inset center when selecting inside the inset.
                center = self._detail_center
                self.table.selectRow(hit["row"])
                self._apply_selection(hit["row"])
                self._detail_center = center
                self._detail_dirty = True
            return True
        if self.chk_detail_inset.isChecked() and event.xdata is not None and event.ydata is not None:
            self._detail_center = (event.xdata, event.ydata)
            self._detail_dirty = True
            self.canvas.draw_idle()
        return False

    def _measure_selected_gap(self):
        if self._gap_worker is not None:
            self._clear_gap()
            self.lbl_gap.setText("Canceling minimum gap search...")
            return
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if len(rows) != 2:
            self.lbl_gap.setText("Select exactly two rows to measure their minimum gap.")
            return
        self.btn_measure_gap.setChecked(False)
        self._clear_gap()
        selected = [self.segments[row] for row in rows]
        pair_count = (len(selected[0].x) // 2) * (len(selected[1].x) // 2)
        if pair_count > 25000:
            worker = _GapWorker(self._gap_version, rows, copy.deepcopy(selected))
            self._gap_worker = worker
            worker.ready.connect(self._minimum_gap_ready)
            worker.failed.connect(self._minimum_gap_failed)
            worker.finished.connect(self._minimum_gap_finished)
            worker.finished.connect(worker.deleteLater)
            self.destroyed.connect(worker.abort.set)
            self.btn_min_gap.setText("Cancel min gap")
            self.btn_min_gap.setEnabled(True)
            self.lbl_gap.setText("Computing minimum gap between the two selected rows...")
            worker.start()
            return
        try:
            result = closest_segment_points(*selected)
        except ValueError as exc:
            self.lbl_gap.setText(str(exc))
            return
        self._minimum_gap_ready(self._gap_version, (rows, result))

    def _minimum_gap_finished(self):
        self._gap_worker = None
        self.btn_min_gap.setText("Min selected gap")
        self.btn_min_gap.setEnabled(len({i.row() for i in self.table.selectedIndexes()}) == 2)

    def _minimum_gap_failed(self, version, message):
        if version == self._gap_version:
            self.lbl_gap.setText(message)

    def _minimum_gap_ready(self, version, payload):
        if version != self._gap_version:
            return
        rows, result = payload
        if result is None:
            self.lbl_gap.setText("Both selected rows need valid primitives to measure.")
            return
        if not all(self._inspection_row_matches(row) for row in rows):
            self.cmb_material_isolation.setCurrentIndex(0)
        self._gap_anchor = None
        self._gap_result = dict(result, first_row=rows[0], second_row=rows[1], kind="Minimum gap")
        self._detail_center = tuple((a + b) / 2 for a, b in zip(result["first_point"], result["second_point"]))
        self._update_gap_label()
        self._render_gap()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _update_gap_label(self):
        result = self._gap_result
        if result is None:
            self.lbl_gap.setText("Ruler: click two boundaries, or select two rows for their minimum gap.")
            return
        first, second = result["first_point"], result["second_point"]
        names = f"{self.segments[result['first_row']].name} / {self.segments[result['second_row']].name}"
        unit = self._geometry_unit_label()
        message = (f"{result['kind']}: {result['distance']:.8g} {unit}  |  "
                   f"dX={second[0]-first[0]:.6g}, dY={second[1]-first[1]:.6g} {unit}  |  {names}")
        if result["kind"] == "Minimum gap" and result["distance"] == 0:
            message += ". Boundaries touch or intersect; use local gap to measure away from the junction."
        self.lbl_gap.setText(message)

    @staticmethod
    def _remove_inspection_artists(artists):
        for artist in artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        artists.clear()

    def _clear_gap(self, *_):
        self._gap_version += 1
        if self._gap_worker is not None:
            self._gap_worker.abort.set()
        self._gap_anchor = None
        self._gap_result = None
        self._remove_inspection_artists(self._gap_artists)
        self._update_gap_label()
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _draw_gap(self, ax):
        artists = []
        points = [self._gap_anchor["point"]] if self._gap_anchor is not None else []
        if self._gap_result is not None:
            points = [self._gap_result["first_point"], self._gap_result["second_point"]]
        if not points:
            return artists
        xs, ys = zip(*points)
        line = Line2D(xs, ys, color="#dc6b00", linewidth=1.5, marker="o",
                      markersize=4, markerfacecolor="white", zorder=19)
        ax.add_line(line)
        artists.append(line)
        if self._gap_result is not None:
            midpoint = tuple((a + b) / 2 for a, b in zip(*points))
            artists.append(ax.annotate(
                f"{self._gap_result['distance']:.6g} {self._geometry_unit_label()}",
                xy=midpoint, xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=8, color="#783700", clip_on=True, zorder=20,
                bbox=dict(facecolor="white", edgecolor="#dc6b00", alpha=.95, pad=2)))
        return artists

    def _render_gap(self):
        self._remove_inspection_artists(self._gap_artists)
        self._gap_artists = self._draw_gap(self.canvas.ax)

    def _on_detail_changed(self, *_):
        self._detail_dirty = True
        self.canvas.draw_idle()

    def _remove_detail(self):
        self._remove_inspection_artists(self._detail_sources)
        if self._detail_ax is not None:
            try:
                self._detail_ax.remove()
            except (ValueError, KeyError):
                pass
        self._detail_ax = None
        self._detail_view = None

    def _inspection_before_load(self):
        self._remove_detail()
        self._clear_gap()
        self.btn_measure_gap.setChecked(False)
        self._detail_center = None

    def _refresh_detail_if_needed(self):
        enabled = self.chk_detail_inset.isChecked() and bool(self.segments)
        if not enabled:
            changed = self._detail_ax is not None
            if changed:
                self._remove_detail()
            return changed
        signature = self._preview_view_key()
        if not self._detail_dirty and self._detail_view == signature:
            return False
        self._detail_dirty = False
        self._detail_view = signature
        main = self.canvas.ax
        if self._detail_center is None:
            center = tuple(sum(limits) / 2 for limits in (main.get_xlim(), main.get_ylim()))
            candidates = []
            for row, segment in enumerate(self.segments):
                if not self._inspection_row_matches(row):
                    continue
                for primitive in self._segment_primitives(segment):
                    if all(math.isfinite(value) for value in primitive):
                        candidates.append(point_to_primitive(center, primitive))
            self._detail_center = min(candidates, key=lambda item: item[1])[0] if candidates else center
        if self._detail_ax is None:
            self._detail_ax = main.inset_axes([.60, .60, .36, .35], zorder=26)
            self._detail_ax.set_navigate(False)
        detail = self._detail_ax
        detail.clear()
        detail.set_navigate(False)
        detail.set_facecolor(main.get_facecolor())
        text = self._plot_theme["text"] if self._plot_theme else "#222222"
        grid = self._plot_theme["grid"] if self._plot_theme else "#999999"
        for spine in detail.spines.values():
            spine.set_color("#dc6b00")
            spine.set_linewidth(1.2)
        detail.tick_params(labelsize=6, colors=text, pad=1)
        detail.ticklabel_format(useOffset=False)
        detail.locator_params(axis="both", nbins=3)
        detail.grid(True, alpha=.22, color=grid)
        zoom = float(self.cmb_detail_zoom.currentData())
        cx, cy = self._detail_center
        width = abs(main.get_xlim()[1] - main.get_xlim()[0]) * .36 / zoom
        height = abs(main.get_ylim()[1] - main.get_ylim()[0]) * .35 / zoom
        detail.set_xlim(cx - width / 2, cx + width / 2)
        detail.set_ylim(cy - height / 2, cy + height / 2)
        detail.set_aspect("equal", adjustable="box")
        detail.set_title(f"Detail {zoom:g}x ({self._geometry_unit_label()})", fontsize=8, color=text, pad=4)
        if self.chk_fill_materials.isChecked():
            self._draw_fill_faces(detail)
        for line in self.segment_lines:
            if line.get_visible():
                detail.add_line(Line2D(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                    linewidth=min(line.get_linewidth(), 1.7), alpha=line.get_alpha(), zorder=3))
        if self.geometry_mode() == "bor":
            detail.axvline(0, color=grid, linestyle="--", linewidth=.8, zorder=2)
        if self.chk_show_impedance.isChecked():
            for artist in self.impedance_artists:
                if isinstance(artist, Line2D):
                    detail.add_line(Line2D(artist.get_xdata(), artist.get_ydata(),
                        color=artist.get_color(), linewidth=2, alpha=artist.get_alpha(), zorder=8))
        self._render_normals(ax=detail)
        self._draw_gap(detail)
        self._remove_inspection_artists(self._detail_sources)
        rectangle = Rectangle((cx - width / 2, cy - height / 2), width, height,
                              facecolor="none", edgecolor="#dc6b00", linewidth=1.1, zorder=20)
        main.add_artist(rectangle)
        connector = ConnectionPatch(xyA=(cx, cy), coordsA="data", axesA=main,
                                    xyB=(0, 0), coordsB="axes fraction", axesB=detail,
                                    color="#dc6b00", linewidth=.7, alpha=.7, zorder=25)
        main.add_artist(connector)
        self._detail_sources.extend((rectangle, connector))
        return True
