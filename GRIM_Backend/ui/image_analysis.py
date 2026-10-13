"""Waterfall axis controls and linked native row/column inspection."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QCheckBox, QLabel, QDoubleSpinBox, QFrame, QHBoxLayout, QSizePolicy
import numpy as np
from matplotlib.ticker import MaxNLocator

from GRIM_Backend.plotting.image_analysis import AXES, cuts
from GRIM_Backend.plotting.modes.common import decimate_line
from GRIM_Backend.plotting.image_grid import line_with_gaps


class SampleCoordinateSpinBox(QDoubleSpinBox):
    """Type physical coordinates; arrow keys step through available samples."""
    samples = None

    def stepBy(self, steps):
        if self.samples is None or not len(self.samples):
            return super().stepBy(steps)
        if steps > 0:
            index = np.searchsorted(self.samples, self.value(), side="right") + steps - 1
        else:
            index = np.searchsorted(self.samples, self.value(), side="left") + steps
        self.setValue(float(self.samples[int(np.clip(index, 0, len(self.samples)-1))]))

    def textFromValue(self, value):
        text = super().textFromValue(value)
        decimal = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(decimal) if decimal in text else text


class ImageAnalysisControls(QObject):
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_x, self.combo_y = QComboBox(), QComboBox()
        for widget in (self.combo_x, self.combo_y):
            for axis in AXES:
                widget.addItem(axis.capitalize(), axis)
        self.combo_y.setCurrentIndex(2)
        self.combo_aggregation = QComboBox()
        self.combo_aggregation.addItem("Separate remaining-axis cuts", "panels")
        self.combo_aggregation.addItem("Mean over remaining axis", "mean_power")
        self.combo_aggregation.setToolTip("All selected cuts are included. Level means use linear display units; phase uses a circular mean. Each polarization remains separate.")
        self._position = None
        self._coordinate_frame = None
        self.controller = None
        self.cut_bar = QFrame()
        self.cut_bar.setObjectName("waterfallCutControls")
        self.cut_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(self.cut_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.check_horizontal = QCheckBox("Horizontal cut")
        self.check_vertical = QCheckBox("Vertical cut")
        self.check_horizontal.setToolTip("Show the horizontal slice at the selected Y coordinate.")
        self.check_vertical.setToolTip("Show the vertical slice at the selected X coordinate.")
        for widget in (self.check_horizontal, self.check_vertical):
            widget.setChecked(True)
            layout.addWidget(widget)
            widget.toggled.connect(lambda: self.changed.emit("heatmap"))
        self.spin_x, self.spin_y = SampleCoordinateSpinBox(), SampleCoordinateSpinBox()
        for label, spin in (("X", self.spin_x), ("Y", self.spin_y)):
            spin.setObjectName("waterfallCut" + label)
            spin.setDecimals(9)
            spin.setRange(-1.e100, 1.e100)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(110)
            spin.setMaximumWidth(165)
            spin.valueChanged.connect(self._position_edited)
            # Enter/focus-out also snaps a typed value that rounds to the
            # existing spin value and therefore emits no valueChanged.
            spin.editingFinished.connect(self._position_edited)
            layout.addWidget(QLabel(label))
            layout.addWidget(spin, 1)
        layout.addStretch(1)
        self.cut_bar.hide()
        self.combo_x.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_x, self.combo_y))
        self.combo_y.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_y, self.combo_x))
        self.combo_aggregation.currentIndexChanged.connect(lambda: self.changed.emit("heatmap"))

    def _axes_changed(self, current, other):
        if current.currentData() == other.currentData():
            blocked = other.blockSignals(True)
            other.setCurrentIndex(next(i for i in range(3) if other.itemData(i) != current.currentData()))
            other.blockSignals(blocked)
        self._position = None
        self._coordinate_frame = None
        self.changed.emit("heatmap")

    def add_rows(self, grid, row):
        for column, (label, widget) in enumerate((("Waterfall X", self.combo_x), ("Y", self.combo_y))):
            grid.addWidget(QLabel(label), row, column*2)
            grid.addWidget(widget, row, column*2+1)
        grid.addWidget(self.combo_aggregation, row+1, 0, 1, 6)
        return row+2

    def axes(self):
        return self.combo_x.currentData(), self.combo_y.currentData()

    def aggregation(self):
        return self.combo_aggregation.currentData()

    def linked_cuts(self):
        return self.horizontal_cuts() or self.vertical_cuts()

    def horizontal_cuts(self):
        return self.check_horizontal.isChecked()

    def vertical_cuts(self):
        return self.check_vertical.isChecked()

    def cut_position(self):
        return self._position

    def bind(self, controller):
        self.controller = controller
        records = [panel[2] for panel in controller.panels]
        record = records[0]
        frame = (record.x_axis, record.y_axis, record.x_unit, record.y_unit)
        if self._coordinate_frame is not None and self._coordinate_frame != frame:
            self._position = None
        self._coordinate_frame = frame
        for spin, axis, unit, values in (
            (self.spin_x, record.x_axis, record.x_unit, np.concatenate([r.x for r in records])),
            (self.spin_y, record.y_axis, record.y_unit, np.concatenate([r.y for r in records])),
        ):
            blocked = spin.blockSignals(True)
            spin.samples = np.unique(values)
            spin.setRange(float(np.min(values)), float(np.max(values)))
            spin.setSuffix(" " + unit if unit else "")
            steps = np.diff(np.unique(values))
            spin.setSingleStep(float(np.min(steps)) if steps.size else 1.)
            spin.setToolTip(f"{axis.capitalize()} ({unit}). Type a coordinate and press Enter to snap to the nearest sample; arrows step through samples. Click the Waterfall to select both coordinates.")
            spin.blockSignals(blocked)
        controller.on_selected = self._selected
        controller.on_disconnect = self._disconnected
        position = self._position or (record.x[len(record.x)//2], record.y[len(record.y)//2])
        controller.update(*position, notify=False)
        self._selected(*controller.position, emit=False)
        self.cut_bar.show()

    def _selected(self, x, y, *, emit=True):
        position = (float(x), float(y))
        changed = position != self._position
        self._position = position
        for spin, value in ((self.spin_x, x), (self.spin_y, y)):
            blocked = spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(blocked)
        if emit and changed:
            self.changed.emit("heatmap_position")

    def _position_edited(self, *_):
        if self.controller is not None:
            self.controller.update(self.spin_x.value(), self.spin_y.value())

    def _disconnected(self):
        self.controller = None
        self.cut_bar.hide()

    def options(self):
        return {"axes": list(self.axes()), "aggregation": self.aggregation(), "linked_cuts": self.linked_cuts(),
                "horizontal_cuts": self.horizontal_cuts(), "vertical_cuts": self.vertical_cuts(),
                "cut_position": list(self._position) if self._position is not None else None}

    def restore_options(self, options):
        if not isinstance(options, dict):
            return
        widgets = (self.combo_x, self.combo_y, self.combo_aggregation, self.check_horizontal, self.check_vertical)
        states = [w.blockSignals(True) for w in widgets]
        try:
            axes = options.get("axes", self.axes())
            if isinstance(axes,(list,tuple)) and len(axes) == 2 and all(isinstance(a,str) and a in AXES for a in axes) and len(set(axes)) == 2:
                for widget, value in zip(widgets[:2], axes):
                    widget.setCurrentIndex(widget.findData(value))
            index = self.combo_aggregation.findData(options.get("aggregation", "panels"))
            if index >= 0:
                self.combo_aggregation.setCurrentIndex(index)
            legacy = bool(options.get("linked_cuts", True))
            self.check_horizontal.setChecked(bool(options.get("horizontal_cuts", legacy)))
            self.check_vertical.setChecked(bool(options.get("vertical_cuts", legacy)))
            self._position = None
            position = options.get("cut_position")
            try:
                if isinstance(position, (list, tuple)) and len(position) == 2 and np.all(np.isfinite(np.asarray(position, float))):
                    self._position = tuple(map(float, position))
            except (ValueError, TypeError, OverflowError):
                pass
            self._coordinate_frame = None
        finally:
            for widget, state in zip(widgets, states):
                widget.blockSignals(state)
        self.changed.emit("heatmap")


class LinkedImageCuts:
    """Linked cuts at actual sample centers, backed by native image records."""
    def __init__(self, canvas, panels, *, can_select=None):
        self.canvas, self.panels = canvas, panels
        self.can_select = can_select
        self.on_selected = self.on_disconnect = None
        self.position = None
        self.artists = []
        units_by_axes = {}
        for _image, cut_axes, record in panels:
            for ax in cut_axes:
                if ax is not None:
                    units_by_axes.setdefault(ax, set()).add(record.value_unit)
        for index,(image_ax, (horizontal, vertical), record) in enumerate(panels,1):
            hline = vline = hcursor = vcursor = None
            if horizontal is not None:
                label = f"{index} ({record.value_unit})" if len(units_by_axes[horizontal]) > 1 else str(index)
                hline, = horizontal.plot([], [], linewidth=1,label=label)
                hcursor = image_ax.axhline(record.y[len(record.y)//2], color="white", linewidth=.7)
                unit = " / ".join(sorted(units_by_axes[horizontal]))
                horizontal.set_ylabel(unit, fontsize=8)
                horizontal.yaxis.set_major_locator(MaxNLocator(3))
            if vertical is not None:
                label = f"{index} ({record.value_unit})" if len(units_by_axes[vertical]) > 1 else str(index)
                vline, = vertical.plot([], [], linewidth=1,label=label)
                vcursor = image_ax.axvline(record.x[len(record.x)//2], color="white", linewidth=.7)
                unit = " / ".join(sorted(units_by_axes[vertical]))
                vertical.set_xlabel(unit, fontsize=8)
                vertical.yaxis.set_major_locator(MaxNLocator(3))
                vertical.xaxis.set_major_locator(MaxNLocator(3))
            self.artists.append((hline, vline, hcursor, vcursor))
        if len(panels)>1:
            for legend_ax in (ax for ax in panels[0][1] if ax is not None):
                legend = legend_ax.legend(ncol=min(4,len(panels)),fontsize=7,loc="upper right")
                for text in legend.get_texts():
                    text.set_color(legend_ax.xaxis.label.get_color())
                legend.get_frame().set_facecolor(legend_ax.get_facecolor())
        self._last = None
        self.connection = canvas.mpl_connect("button_press_event", self.on_click)
        if panels:
            record = panels[0][2]
            self.update(record.x[len(record.x)//2], record.y[len(record.y)//2])

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.panels = []
        self.artists = []
        if self.on_disconnect is not None:
            self.on_disconnect()
        self.on_selected = self.on_disconnect = self.can_select = None

    def on_click(self, event):
        axes = [p[0] for p in self.panels]
        if event.button != 1 or event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        # Cleared figures can still own this controller until their next
        # heatmap; never read a disposed native snapshot from a stale axes.
        if event.inaxes not in self.canvas.figure.axes:
            return
        if (getattr(getattr(self.canvas, "toolbar", None), "mode", "")
                or self.canvas.widgetlock.locked()
                or (self.can_select is not None and not self.can_select())):
            return
        self.update(event.xdata, event.ydata, anchor=axes.index(event.inaxes))

    def update(self, x, y, *, anchor=0, notify=True):
        if not self.panels or not np.all(np.isfinite([x, y])):
            return
        # Anchor typed coordinates to the first panel; clicks anchor to the
        # clicked panel. Other datasets snap independently to that position.
        source = self.panels[anchor][2]
        x = float(source.x[int(np.argmin(abs(source.x-x)))])
        y = float(source.y[int(np.argmin(abs(source.y-y)))])
        self.position = (x, y)
        positions = tuple((int(np.argmin(abs(p[2].x-x))), int(np.argmin(abs(p[2].y-y)))) for p in self.panels)
        if notify and self.on_selected is not None:
            self.on_selected(x, y)
        if positions == self._last:
            return
        self._last = positions
        selected_x, selected_y = [], []
        for (image_ax, (horizontal, vertical), record), artists in zip(self.panels, self.artists):
            xs, row, ys, column, px, py = cuts(record, x, y)
            selected_x.append(px)
            selected_y.append(py)
            hline, vline, hcursor, vcursor = artists
            if hline is not None:
                hx,hy,_ = decimate_line(*line_with_gaps(xs,row,breaks=record.metadata.get("x_cut_breaks")))
                hline.set_data(hx, hy)
                hcursor.set_ydata([py,py])
            if vline is not None:
                vy,vx,_ = decimate_line(*line_with_gaps(ys,column,breaks=record.metadata.get("y_cut_breaks")))
                vline.set_data(vx, vy)
                vcursor.set_xdata([px,px])
        horizontal, vertical = self.panels[0][1]
        record = self.panels[0][2]
        for ax, values, direction, axis, unit in (
            (horizontal, selected_y, "Horizontal", record.y_axis, record.y_unit),
            (vertical, selected_x, "Vertical", record.x_axis, record.x_unit),
        ):
            if ax is None:
                continue
            ax.relim()
            ax.autoscale_view(scalex=ax is vertical, scaley=ax is horizontal)
            label = f"{min(values):.6g}" if min(values) == max(values) else f"{min(values):.6g}–{max(values):.6g} (per panel)"
            separator = "\n" if ax is vertical else " · "
            ax.set_title(f"{direction} cuts{separator}{axis} = {label} {unit}", fontsize=8,
                         color=ax.xaxis.label.get_color(),pad=2)
        self.canvas.draw_idle()
