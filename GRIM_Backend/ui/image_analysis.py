"""Heatmap axis controls and linked native row/column inspection."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QCheckBox, QLabel
import numpy as np
from matplotlib.ticker import MaxNLocator

from GRIM_Backend.plotting.image_analysis import AXES, cuts
from GRIM_Backend.plotting.modes.common import decimate_line


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
        self.check_cuts = QCheckBox("Linked horizontal / vertical cuts")
        self.check_cuts.setChecked(True)
        self.combo_x.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_x, self.combo_y))
        self.combo_y.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_y, self.combo_x))
        self.combo_aggregation.currentIndexChanged.connect(lambda: self.changed.emit("heatmap"))
        self.check_cuts.toggled.connect(lambda: self.changed.emit("heatmap"))

    def _axes_changed(self, current, other):
        if current.currentData() == other.currentData():
            blocked = other.blockSignals(True)
            other.setCurrentIndex(next(i for i in range(3) if other.itemData(i) != current.currentData()))
            other.blockSignals(blocked)
        self.changed.emit("heatmap")

    def add_rows(self, grid, row):
        for column, (label, widget) in enumerate((("Waterfall X", self.combo_x), ("Y", self.combo_y))):
            grid.addWidget(QLabel(label), row, column*2)
            grid.addWidget(widget, row, column*2+1)
        grid.addWidget(self.combo_aggregation, row+1, 0, 1, 3)
        grid.addWidget(self.check_cuts, row+1, 3, 1, 3)
        return row+2

    def axes(self):
        return self.combo_x.currentData(), self.combo_y.currentData()

    def aggregation(self):
        return self.combo_aggregation.currentData()

    def linked_cuts(self):
        return self.check_cuts.isChecked()

    def options(self):
        return {"axes": list(self.axes()), "aggregation": self.aggregation(), "linked_cuts": self.linked_cuts()}

    def restore_options(self, options):
        if not isinstance(options, dict):
            return
        widgets = (self.combo_x, self.combo_y, self.combo_aggregation, self.check_cuts)
        states = [w.blockSignals(True) for w in widgets]
        try:
            axes = options.get("axes", self.axes())
            if isinstance(axes,(list,tuple)) and len(axes) == 2 and all(isinstance(a,str) and a in AXES for a in axes) and len(set(axes)) == 2:
                for widget, value in zip(widgets[:2], axes):
                    widget.setCurrentIndex(widget.findData(value))
            index = self.combo_aggregation.findData(options.get("aggregation", "panels"))
            if index >= 0:
                self.combo_aggregation.setCurrentIndex(index)
            self.check_cuts.setChecked(bool(options.get("linked_cuts", True)))
        finally:
            for widget, state in zip(widgets, states):
                widget.blockSignals(state)
        self.changed.emit("heatmap")


class LinkedImageCuts:
    """Linked cuts at actual sample centers, backed by native image records."""
    def __init__(self, canvas, panels):
        self.canvas, self.panels = canvas, panels
        self.artists = []
        for index,(image_ax, (horizontal, vertical), record) in enumerate(panels,1):
            hline, = horizontal.plot([], [], linewidth=1,label=str(index))
            vline, = vertical.plot([], [], linewidth=1,label=str(index))
            hcursor = image_ax.axhline(record.y[len(record.y)//2], color="white", linewidth=.7)
            vcursor = image_ax.axvline(record.x[len(record.x)//2], color="white", linewidth=.7)
            horizontal.set_ylabel(record.value_unit, fontsize=8)
            vertical.set_xlabel(record.value_unit, fontsize=8)
            horizontal.yaxis.set_major_locator(MaxNLocator(3))
            vertical.yaxis.set_major_locator(MaxNLocator(3))
            vertical.xaxis.set_major_locator(MaxNLocator(3))
            self.artists.append((hline, vline, hcursor, vcursor))
        if len(panels)>1:
            legend=panels[0][1][0].legend(ncol=min(4,len(panels)),fontsize=7,loc="upper right")
            for text in legend.get_texts():
                text.set_color(panels[0][1][0].xaxis.label.get_color())
            legend.get_frame().set_facecolor(panels[0][1][0].get_facecolor())
        self._last = None
        self.connection = canvas.mpl_connect("motion_notify_event", self.on_motion)
        if panels:
            record = panels[0][2]
            self.update(record.x[len(record.x)//2], record.y[len(record.y)//2])

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.panels = []
        self.artists = []

    def on_motion(self, event):
        if event.inaxes not in [p[0] for p in self.panels] or event.xdata is None or event.ydata is None:
            return
        # Cleared figures can still own this controller until their next
        # heatmap; never read a disposed native snapshot from a stale axes.
        if event.inaxes not in self.canvas.figure.axes:
            return
        self.update(event.xdata, event.ydata)

    def update(self, x, y):
        positions = tuple((int(np.argmin(abs(p[2].x-x))), int(np.argmin(abs(p[2].y-y)))) for p in self.panels)
        if positions == self._last:
            return
        self._last = positions
        for (image_ax, (horizontal, vertical), record), artists in zip(self.panels, self.artists):
            xs, row, ys, column, px, py = cuts(record, x, y)
            hline, vline, hcursor, vcursor = artists
            hx,hy,_ = decimate_line(xs,row)
            vy,vx,_ = decimate_line(ys,column)
            hline.set_data(hx, hy)
            vline.set_data(vx, vy)
            hcursor.set_ydata([py,py])
            vcursor.set_xdata([px,px])
            horizontal.relim(); horizontal.autoscale_view(scalex=False)
            vertical.relim(); vertical.autoscale_view(scaley=False)
            horizontal.set_title(f"Horizontal cuts · {record.y_axis} ≈ {py:.6g} {record.y_unit}", fontsize=8,color=horizontal.xaxis.label.get_color(),pad=2)
            vertical.set_title(f"Vertical cuts · {record.x_axis} ≈ {px:.6g}", fontsize=8,color=vertical.xaxis.label.get_color(),pad=2)
        self.canvas.draw_idle()
