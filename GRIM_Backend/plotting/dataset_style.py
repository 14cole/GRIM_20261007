"""Dataset-level interaction and view styles for the Plotting canvas."""

from __future__ import annotations

from matplotlib import patheffects
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QInputDialog, QMenu


HIGHLIGHT_COLOR = "#d3d3d3"
LINE_TYPES = (("Solid", "-"), ("Dashed", "--"), ("Dotted", ":"), ("Dash-dot", "-."))
SCATTER_SYMBOLS = (
    ("Circle", "o"), ("Square", "s"), ("Triangle up", "^"),
    ("Triangle down", "v"), ("Diamond", "D"), ("Star", "*"),
    ("Plus", "+"), ("Cross", "x"), ("Pentagon", "p"), ("Hexagon", "h"),
)
_LINE_PROPERTIES = (
    "color", "linewidth", "linestyle", "marker", "markersize",
    "markerfacecolor", "markeredgecolor", "markeredgewidth",
)
# Plot-item key shared by every artist of one PBP envelope, so a band is
# selected, highlighted, and removed as one item alongside dataset curves.
# PBP by group draws one band per group name; the ungrouped band uses "".
def pbp_band_key(group: str | None = None) -> tuple[str, str]:
    return ("pbp_band", str(group or ""))


def is_pbp_band_key(key) -> bool:
    return isinstance(key, tuple) and len(key) == 2 and key[0] == "pbp_band"


PBP_BAND_KEY = pbp_band_key()


def _legend_label(artist) -> str | None:
    label = artist.get_label()
    if isinstance(label, str) and label and not label.startswith("_"):
        return label
    return None


class DatasetPlotStyleMixin:
    def _dataset_plot_key(self, dataset):
        reference_for = getattr(self, "_python_reference_for_dataset", None)
        reference = reference_for(dataset) if callable(reference_for) else None
        # GUI row IDs survive renaming, sorting, and edits to dataset values.
        return reference.dataset_id if reference is not None else id(dataset)

    def _register_dataset_line(self, line, dataset) -> None:
        self._register_plot_line(line, self._dataset_plot_key(dataset))

    def _register_plot_line(self, line, key) -> None:
        line._grim_dataset_key = key
        line._grim_base_style = {
            name: getattr(line, f"get_{name}")() for name in _LINE_PROPERTIES
        }
        line._grim_base_zorder = line.get_zorder()
        line._grim_base_path_effects = line.get_path_effects()
        line.set_pickradius(6)
        self._apply_dataset_line_style(line)

    def _apply_dataset_line_style(self, line) -> None:
        key = getattr(line, "_grim_dataset_key", None)
        if key is None:
            return
        base = line._grim_base_style
        style = getattr(self, "_dataset_plot_styles", {}).get(key, {})
        line.set(**base)
        for name in ("color", "linewidth", "linestyle", "markersize"):
            if name in style:
                getattr(line, f"set_{name}")(style[name])
        if "color" in style:
            line.set_markerfacecolor(style["color"])
            line.set_markeredgecolor(style["color"])
        # Marker-only Line2D artists keep the same data, picking, Hold, and
        # export behavior as curves, with no connecting segments in scatter mode.
        if style.get("kind", "line") == "scatter":
            line.set_linestyle("None")
            line.set_marker(style.get("marker", "o"))
        selected = key in getattr(self, "_highlighted_plot_datasets", set())
        effects = line._grim_base_path_effects
        if selected:
            effects = [
                patheffects.Stroke(
                    linewidth=max(line.get_linewidth(), line.get_markeredgewidth()) + 5,
                    foreground=HIGHLIGHT_COLOR,
                ),
                patheffects.Normal(),
            ]
        line.set_path_effects(effects)
        line.set_zorder(line._grim_base_zorder + (10 if selected else 0))

    def _plotting_style_figure(self):
        contexts = getattr(self, "_plot_contexts", {})
        if "plotting" in contexts:
            return contexts["plotting"].plot_figure
        return self.plot_figure

    def _sync_dataset_legend(self, ax, text_color=None) -> None:
        legend = ax.get_legend()
        if legend is None:
            return
        handles, _labels = ax.get_legend_handles_labels()
        proxies = legend.legend_handles
        for source, proxy, text in zip(handles, proxies, legend.get_texts()):
            key = getattr(source, "_grim_dataset_key", None)
            if key is None:
                continue
            if isinstance(source, Line2D) and isinstance(proxy, Line2D):
                for name in _LINE_PROPERTIES:
                    getattr(proxy, f"set_{name}")(getattr(source, f"get_{name}")())
                proxy.set_path_effects(source.get_path_effects())
            normal_color = text_color or self._current_plot_text()
            selected = key in getattr(self, "_highlighted_plot_datasets", set())
            text.set_bbox(
                dict(facecolor=HIGHLIGHT_COLOR, edgecolor="none", pad=1.5)
                if selected else None
            )
            text.set_color("#202020" if selected else normal_color)

    def _refresh_dataset_plot_styles(self) -> None:
        figure = self._plotting_style_figure()
        for ax in figure.axes:
            for line in ax.lines:
                self._apply_dataset_line_style(line)
            self._sync_dataset_legend(ax)
        figure.canvas.draw_idle()

    def _highlight_plot_dataset(self, key) -> None:
        self._highlighted_plot_datasets = set() if key is None else {key}
        self._refresh_dataset_plot_styles()

    def _set_dataset_plot_style(self, keys, **changes) -> None:
        styles = getattr(self, "_dataset_plot_styles", None)
        if styles is None:
            styles = self._dataset_plot_styles = {}
        for key in keys:
            if changes:
                styles.setdefault(key, {}).update(changes)
            else:
                styles.pop(key, None)
        self._refresh_dataset_plot_styles()

    def _dataset_line_at_event(self, event):
        """Hit-test legend entries first, then visible curves and scatter points."""
        if getattr(self, "_active_plot_tab", "plotting") != "plotting":
            return None
        if event.canvas is not self.plot_canvas:
            return None
        for ax in self.plot_figure.axes:
            legend = ax.get_legend()
            if legend is not None and legend.get_visible() and legend.contains(event)[0]:
                handles, _labels = ax.get_legend_handles_labels()
                for source, proxy, text in zip(
                    handles, legend.legend_handles, legend.get_texts()
                ):
                    if getattr(source, "_grim_dataset_key", None) is None:
                        continue
                    if text.contains(event)[0] or (proxy is not None and proxy.contains(event)[0]):
                        return source
                # A click on the legend frame must not pick a curve behind it.
                return None
        if event.inaxes is None:
            return None
        lines = sorted(event.inaxes.lines, key=lambda line: line.get_zorder(), reverse=True)
        for line in lines:
            if (
                getattr(line, "_grim_dataset_key", None) is not None
                and line.get_visible()
                and line.contains(event)[0]
            ):
                return line
        return None

    def _dataset_line_at_canvas_position(self, pos):
        x, y = self.plot_canvas.mouseEventCoords(pos)
        event = MouseEvent("button_press_event", self.plot_canvas, x, y, button=3)
        return self._dataset_line_at_event(event)

    def _add_dataset_plot_style_menu(self, menu, keys, *, line=None) -> None:
        """Add controls for one clicked dataset or the selected dataset rows."""
        keys = tuple(keys)
        styles = getattr(self, "_dataset_plot_styles", {})
        current = styles.get(keys[0], {})
        base = getattr(line, "_grim_base_style", {})

        def change(**values):
            self._set_dataset_plot_style(keys, **values)

        def choices(title, property_name, options, value):
            submenu = menu.addMenu(title)
            for label, option in options:
                action = submenu.addAction(label)
                action.setCheckable(True)
                action.setChecked(option == value)
                action.triggered.connect(
                    lambda _checked=False, option=option: change(**{property_name: option})
                )

        choices("Plot as", "kind", (("Line", "line"), ("Scatter", "scatter")), current.get("kind", "line"))
        choices("Line type", "linestyle", LINE_TYPES, current.get("linestyle", base.get("linestyle", "-")))

        def choose_width():
            value, accepted = QInputDialog.getDouble(
                self, "Dataset line width", "Line width (points):",
                current.get("linewidth", base.get("linewidth", 1.5)), 0.1, 20.0, 1,
            )
            if accepted:
                change(linewidth=value)

        menu.addAction("Line width…", choose_width)

        def choose_color():
            color = QColorDialog.getColor(
                QColor(to_hex(current.get("color", base.get("color", "#1f77b4")))),
                self, "Dataset plot color",
            )
            if color.isValid():
                change(color=color.name())

        menu.addAction("Color…", choose_color)
        choices("Scatter symbol", "marker", SCATTER_SYMBOLS, current.get("marker", "o"))

        def choose_size():
            value, accepted = QInputDialog.getDouble(
                self, "Scatter symbol size", "Symbol size (points):",
                current.get("markersize", base.get("markersize", 6.0)), 1.0, 40.0, 1,
            )
            if accepted:
                change(markersize=value)

        menu.addAction("Scatter symbol size…", choose_size)
        menu.addSeparator()
        menu.addAction("Reset plot style", lambda: self._set_dataset_plot_style(keys))

    def _show_dataset_plot_style_menu(self, line, global_pos) -> None:
        self._dataset_plot_menu(line).exec(global_pos)

    def _dataset_plot_menu(self, line) -> QMenu:
        """Build the right-click menu for one curve, legend entry, or PBP band."""
        key = line._grim_dataset_key
        self._highlight_plot_dataset(key)
        menu = QMenu(self)
        menu.addSection(self._plot_item_name(key))
        if not is_pbp_band_key(key):
            self._add_dataset_plot_style_menu(menu, [key], line=line)
            menu.addSeparator()
        remove = menu.addAction(
            "Remove from plot\tDel", lambda: self._remove_plot_dataset(key)
        )
        remove.setEnabled(self._plot_item_removal_blocked_reason() is None)
        return menu

    def _plot_item_artists(self, key) -> list:
        return [
            artist
            for ax in self.plot_figure.axes
            for artist in (*ax.lines, *ax.collections)
            if getattr(artist, "_grim_dataset_key", None) == key
        ]

    def _plot_item_name(self, key, artists=None) -> str:
        for artist in self._plot_item_artists(key) if artists is None else artists:
            label = _legend_label(artist)
            if label is not None:
                return label.split(" | ", 1)[0]
        return "PBP band" if is_pbp_band_key(key) else "Dataset"

    def _remove_plot_item_artists(self, key) -> list:
        """Detach one dataset's curves (or the PBP band) from the canvas."""
        removed = []
        for ax in self.plot_figure.axes:
            artists = [
                artist for artist in (*ax.lines, *ax.collections)
                if getattr(artist, "_grim_dataset_key", None) == key
            ]
            for artist in artists:
                artist.remove()
            legend = ax.get_legend()
            if legend is not None and any(_legend_label(a) for a in artists):
                # The legend is rebuilt from the remaining artists.
                legend.remove()
            removed.extend(artists)
        return removed

    def _plot_item_removal_blocked_reason(self) -> str | None:
        if getattr(self, "last_plot_mode", None) == "compare":
            return (
                "RF Compare always shows its two datasets together; select a "
                "different pair and plot again."
            )
        return None

    def _remove_plot_dataset(self, key) -> bool:
        """Remove one dataset (or the PBP band) from the Plotting canvas only.

        The dataset table selection is left alone, so plotting again brings
        the dataset back.
        """
        if key is None or getattr(self, "_active_plot_tab", "plotting") != "plotting":
            return False
        blocked = self._plot_item_removal_blocked_reason()
        if blocked is not None:
            self.status.showMessage(blocked)
            return False
        removed = self._remove_plot_item_artists(key)
        if not removed:
            return False
        name = self._plot_item_name(key, removed)
        self._highlighted_plot_datasets = (
            set(getattr(self, "_highlighted_plot_datasets", set())) - {key}
        )
        figure = self.plot_figure
        figure._grim_held_phase_datasets = [
            entry for entry in getattr(figure, "_grim_held_phase_datasets", [])
            if self._dataset_plot_key(entry[1]) != key
        ]
        self._drop_dataset_from_python_plot_spec(key)
        self._restore_plot_markers()
        self._update_legend_visibility()
        self._maybe_autoscale()
        self.plot_canvas.draw_idle()
        self.status.showMessage(f"Removed {name} from the plot. Plot again to restore it.")
        return True

    def _on_plot_key_press(self, event) -> None:
        if getattr(event, "canvas", None) is not self.plot_canvas:
            return
        if self._on_marker_key(event):
            return
        if getattr(event, "key", None) not in ("delete", "backspace"):
            return
        keys = list(getattr(self, "_highlighted_plot_datasets", set()))
        removed = [key for key in keys if self._remove_plot_dataset(key)]
        if not removed and self._plot_item_removal_blocked_reason() is None:
            self.status.showMessage(
                "Click a curve or legend entry to select it, then press Delete "
                "to remove it from the plot."
            )
