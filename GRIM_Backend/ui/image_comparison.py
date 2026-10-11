"""Nonmodal frozen A/B image inspection with linked views, cuts and ROI."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QFileDialog, QScrollArea, QWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.widgets import RectangleSelector

from GRIM_Backend.plotting.image_analysis import (records_from_figure, compare_records,
    summarize_roi, ordered, comparison_advisories)
from GRIM_Backend.plotting.modes.common import coordinate_edges, finite_data_limits, decimate_line


class ImageComparisonDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner if hasattr(owner, "winId") else None)
        self.owner = owner
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Images A/B — frozen images and linked inspection")
        from GRIM_Backend.ui.widgets import initial_window_size
        screen = self.screen()
        self.resize(initial_window_size(screen.availableGeometry().size(), QSize(1150,860)))
        self.a = self.b = None
        self.result = None
        self.roi_bounds = None
        self._selectors = []
        outer = QVBoxLayout(self)
        controls = QWidget()
        layout = QVBoxLayout(controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        palette = getattr(owner,"application_palette",{})
        background = palette.get("panel_bg", "#ffffff")
        foreground = palette.get("text", "#111111")
        scroll.setStyleSheet(f"QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {background}; color: {foreground}; }} QLabel,QCheckBox {{ color: {foreground}; }}")
        scroll.setMaximumHeight(240)
        scroll.setMinimumHeight(140)
        outer.addWidget(scroll)
        source_row = QHBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.setMinimumWidth(200)
        source_row.addWidget(QLabel("Current image"))
        source_row.addWidget(self.combo_source, 1)
        for text, callback in (("Refresh panels", self.refresh_sources), ("Capture as A", lambda: self.capture("a")),
                ("Capture as B", lambda: self.capture("b")), ("Swap A/B", self.swap)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            source_row.addWidget(button)
        layout.addLayout(source_row)
        self.names = QLabel("Capture A, change the dataset or image settings, then capture B. Source data is preserved.")
        self.names.setWordWrap(True)
        layout.addWidget(self.names)
        options = QHBoxLayout()
        self.combo_policy = QComboBox()
        for label, value in (("Matching sample centers", "exact"), ("Nearest neighbor on overlap", "nearest"), ("Linear intensity on overlap", "linear")):
            self.combo_policy.addItem(label, value)
        self.combo_mode = QComboBox()
        for label, value in (("Physical A − B", "physical"), ("Relative to each peak", "relative"), ("Displayed numeric A − B", "displayed")):
            self.combo_mode.addItem(label, value)
        self.check_as_supplied = QCheckBox("Compare coordinates as supplied")
        self.check_as_supplied.setToolTip("For unlike axis meanings, use the numeric coordinates without assuming a physical frame conversion. Frame/recipe metadata differences are always shown as advisories.")
        for widget in (QLabel("Grid"), self.combo_policy, self.combo_mode, self.check_as_supplied):
            options.addWidget(widget)
        layout.addLayout(options)
        view_row = QHBoxLayout()
        self.combo_view = QComboBox()
        for label, value in (("A / B with signed difference", "side"), ("Blink A ↔ B", "blink"), ("Swipe A | B", "swipe")):
            self.combo_view.addItem(label, value)
        self.swipe = QSlider(Qt.Horizontal)
        self.swipe.setRange(0,100)
        self.swipe.setValue(50)
        self.check_roi = QCheckBox("Draw ROI on an image")
        clear_roi = QPushButton("Full image ROI")
        clear_roi.clicked.connect(self.clear_roi)
        export = QPushButton("Export ROI summary")
        export.clicked.connect(self.export_summary)
        for widget in (self.combo_view, self.swipe, self.check_roi, clear_roi, export):
            view_row.addWidget(widget)
        layout.addLayout(view_row)
        self.advisory = QLabel()
        self.advisory.setWordWrap(True)
        self.advisory.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.advisory)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(300,240)
        outer.addWidget(NavigationToolbar2QT(self.canvas, self))
        outer.addWidget(self.canvas, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.summary)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._blink)
        self._blink_b = False
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        for combo in (self.combo_policy, self.combo_mode):
            combo.currentIndexChanged.connect(self.render)
        self.check_as_supplied.toggled.connect(self.render)
        self.combo_view.currentIndexChanged.connect(self._view_changed)
        self.swipe.valueChanged.connect(self._update_primary)
        self.check_roi.toggled.connect(self._roi_toggled)
        self.refresh_sources()
        if self.combo_source.count():
            self.capture("a")
        if self.combo_source.count() > 1:
            records = records_from_figure(self.owner)
            source_name = records[0].metadata.get("source_name")
            second = next((i for i,r in enumerate(records[1:],1) if r.metadata.get("source_name") != source_name),1)
            self.combo_source.setCurrentIndex(second)
            self.capture("b")

    def refresh_sources(self):
        index = self.combo_source.currentIndex()
        self.combo_source.clear()
        for record in records_from_figure(self.owner):
            self.combo_source.addItem(record.name)
        if self.combo_source.count():
            self.combo_source.setCurrentIndex(max(0, min(index, self.combo_source.count()-1)))

    def capture(self, side):
        index = max(0, self.combo_source.currentIndex())
        self.refresh_sources()
        records = records_from_figure(self.owner)
        if not records:
            self.advisory.setText("Render a Waterfall, ISAR image, or Azimuth–Down-Range image, then refresh and capture it here.")
            return
        try:
            record = records[min(index, len(records)-1)].frozen()
        except (OSError, MemoryError, ValueError) as exc:
            self.advisory.setText(f"Image capture unavailable: {exc}")
            return
        old = getattr(self, side)
        setattr(self, side, record)
        if old is not None:
            old.close()
        self.roi_bounds = None
        self.render()

    def swap(self):
        self.a, self.b = self.b, self.a
        self.roi_bounds = None
        self.render()

    @staticmethod
    def _mesh(ax, x, y, values, **kwargs):
        return ax.pcolormesh(coordinate_edges(x), coordinate_edges(y), np.ma.masked_invalid(values), shading="flat", **kwargs)

    def render(self, *_args):
        self.timer.stop()
        for selector in self._selectors:
            selector.disconnect_events()
        self._selectors = []
        self.result = None
        self.figure.clear()
        self.image_axes = []
        self.names.setText(f"A: {self.a.name if self.a else 'not captured'}\nB: {self.b.name if self.b else 'not captured'}")
        if self.a is None or self.b is None:
            self.advisory.setText("Capture both A and B. Captures stay fixed while the main image changes.")
            self.canvas.draw_idle()
            return
        try:
            self.result = compare_records(self.a, self.b, policy=self.combo_policy.currentData(),
                mode=self.combo_mode.currentData(), as_supplied=self.check_as_supplied.isChecked())
        except (MemoryError,OSError) as exc:
            self.advisory.setText(f"Comparison storage or allocation failed: {exc}. Free memory or capture a smaller input selection and retry.")
            self.summary.setText("The frozen source captures are preserved.")
            self.canvas.draw_idle()
            return
        except ValueError as exc:
            self.advisory.setText(f"Quantitative difference unavailable: {exc}\n" + " · ".join(comparison_advisories(self.a,self.b)))
            # Inspect both supplied images even when a numeric difference
            # needs a different coordinate/ordinate choice.
            for i, record in enumerate((self.a,self.b)):
                ax = self.figure.add_subplot(1,2,i+1)
                try:
                    x,y,values = ordered(record)
                except (ValueError,MemoryError,OSError) as image_error:
                    ax.text(.5,.5,str(image_error),transform=ax.transAxes,ha="center",wrap=True)
                    continue
                mesh = self._mesh(ax,x,y,values,cmap="viridis")
                self.figure.colorbar(mesh, ax=ax, label=record.value_unit)
                ax.set_title(("A: " if i == 0 else "B: ") + record.name)
                ax.set_xlabel(f"{record.x_axis} ({record.x_unit})")
                ax.set_ylabel(f"{record.y_axis} ({record.y_unit})")
            self.summary.setText("Separate native axes/scales are shown until an explicit comparison policy is selected.")
            self.canvas.draw_idle()
            return
        result = self.result
        x,y,a,b,delta = (result[k] for k in ("x","y","a","b","difference"))
        limits = finite_data_limits((a,b)) or (0.,1.)
        if limits[0] == limits[1]:
            limits = (limits[0]-.5,limits[1]+.5)
        scale = float(np.nanmax(abs(delta))) if np.any(np.isfinite(delta)) else 1.
        scale = max(scale, 1.e-12)
        grid = self.figure.add_gridspec(2,3, height_ratios=[3,2])
        self.image_axes = [self.figure.add_subplot(grid[0,i]) for i in range(3)]
        for ax in self.image_axes[1:]:
            ax.sharex(self.image_axes[0]); ax.sharey(self.image_axes[0])
        self.mesh_a = self._mesh(self.image_axes[0],x,y,a,cmap="viridis",norm=Normalize(*limits))
        mesh_b = self._mesh(self.image_axes[1],x,y,b,cmap="viridis",norm=self.mesh_a.norm)
        mesh_d = self._mesh(self.image_axes[2],x,y,delta,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0.,vmin=-scale,vmax=scale))
        for ax,title in zip(self.image_axes,("A","B","Signed A − B")):
            ax.set_title(title)
            ax.set_xlabel(f"{self.a.x_axis} ({self.a.x_unit})")
            ax.set_ylabel(f"{self.a.y_axis} ({self.a.y_unit})")
            ax.set_xlim(coordinate_edges(x)[[0,-1]])
            ax.set_ylim(coordinate_edges(y)[[0,-1]])
        level_unit = ("relative level (dB)" if self.combo_mode.currentData() == "relative" else
            ("deg" if self.a.phase else result['level_unit']))
        if self.combo_mode.currentData() == "displayed":
            level_unit = f"supplied A: {self.a.value_unit}; B: {self.b.value_unit}"
        self.figure.colorbar(mesh_b,ax=self.image_axes[:2],label=level_unit,shrink=.85)
        self.figure.colorbar(mesh_d,ax=self.image_axes[2],label=result['unit'],shrink=.85)
        self.horizontal = self.figure.add_subplot(grid[1,:2])
        self.vertical = self.figure.add_subplot(grid[1,2])
        self.hlines = self.horizontal.plot([],[],label="A") + self.horizontal.plot([],[],label="B")
        self.vlines = self.vertical.plot([],[],label="A") + self.vertical.plot([],[],label="B")
        self.horizontal.legend(); self.vertical.legend()
        self.horizontal.set_xlabel(f"{self.a.x_axis} ({self.a.x_unit})")
        self.horizontal.set_ylabel(level_unit)
        self.vertical.set_ylabel(f"{self.a.y_axis} ({self.a.y_unit})")
        self.vertical.set_xlabel(level_unit)
        self.cursors = [(ax.axvline(x[len(x)//2], color="white",linewidth=.7), ax.axhline(y[len(y)//2], color="white",linewidth=.7)) for ax in self.image_axes]
        self._selectors = [RectangleSelector(ax,self._on_roi,button=[1],useblit=False,interactive=True) for ax in self.image_axes]
        self._roi_toggled(self.check_roi.isChecked())
        self.advisory.setText("Locked coordinates and A/B color scale on overlapping centers. " +
            (("Interpolation uses supplied numeric values. " if self.combo_mode.currentData() == "displayed" else "Interpolation uses physical linear intensity (unit phasors for phase); missing corners stay missing. ") if result['policy']=='linear' else "") +
            " · ".join(result["notes"]))
        self._last_cursor = None
        self.update_cuts(x[len(x)//2],y[len(y)//2])
        self.update_summary()
        self._view_changed()
        self.canvas.draw_idle()

    def _view_changed(self, *_args):
        self.timer.stop()
        mode = self.combo_view.currentData()
        self.swipe.setEnabled(mode == "swipe")
        if self.result is not None:
            if mode == "blink":
                self.timer.start()
            self._update_primary()

    def _blink(self):
        self._blink_b = not self._blink_b
        self._update_primary()

    def _update_primary(self, *_args):
        if self.result is None:
            return
        mode = self.combo_view.currentData()
        a,b = self.result['a'],self.result['b']
        title, values = "A", a
        if mode == "blink":
            title, values = ("Blink: B",b) if self._blink_b else ("Blink: A",a)
        elif mode == "swipe":
            x = self.result['x']
            split = x[0] + (x[-1]-x[0])*self.swipe.value()/100.
            values = np.where(x[None,:] <= split,a,b)
            title = f"Swipe A | B at {split:.6g} {self.a.x_unit}"
        self.mesh_a.set_array(np.ma.masked_invalid(values))
        self.image_axes[0].set_title(title)
        self.canvas.draw_idle()

    def on_motion(self,event):
        if self.result is not None and event.inaxes in self.image_axes and event.xdata is not None and event.ydata is not None:
            self.update_cuts(event.xdata,event.ydata)

    def update_cuts(self,x,y):
        result = self.result
        ix,iy = int(np.argmin(abs(result['x']-x))),int(np.argmin(abs(result['y']-y)))
        if (ix,iy) == self._last_cursor:
            return
        self._last_cursor = ix,iy
        for index,key in enumerate(('a','b')):
            hx,hy,_ = decimate_line(result['x'],result[key][iy,:])
            vy,vx,_ = decimate_line(result['y'],result[key][:,ix])
            self.hlines[index].set_data(hx,hy)
            self.vlines[index].set_data(vx,vy)
        for v,h in self.cursors:
            v.set_xdata([result['x'][ix]]*2); h.set_ydata([result['y'][iy]]*2)
        for ax in (self.horizontal,self.vertical):
            ax.relim(); ax.autoscale_view()
        self.horizontal.set_title(f"Row at {self.a.y_axis} = {result['y'][iy]:.6g} {self.a.y_unit}")
        self.vertical.set_title(f"Column at {self.a.x_axis} = {result['x'][ix]:.6g} {self.a.x_unit}")
        self.canvas.draw_idle()

    def _roi_toggled(self,enabled):
        for selector in self._selectors:
            selector.set_active(enabled)

    def _on_roi(self,start,stop):
        if None in (start.xdata,start.ydata,stop.xdata,stop.ydata):
            return
        self.roi_bounds = (min(start.xdata,stop.xdata),max(start.xdata,stop.xdata),min(start.ydata,stop.ydata),max(start.ydata,stop.ydata))
        for selector in self._selectors:
            selector.extents = self.roi_bounds
        self.update_summary()
        self.canvas.draw_idle()

    def clear_roi(self):
        self.roi_bounds = None
        for selector in self._selectors:
            selector.set_visible(False)
        self.update_summary()
        self.canvas.draw_idle()

    def update_summary(self):
        if self.result is None:
            return
        try:
            stats = summarize_roi(self.result,self.roi_bounds)
            self.summary.setText(f"ROI: {stats['pixels']:,} finite cells / {stats['selected_cells']:,}; {stats['excluded_cells']:,} excluded. "
                f"A−B {stats['mean_definition']} mean {stats['mean_delta']:.6g}; RMS {stats['rms_delta']:.6g}; "
                f"min {stats['minimum_delta']:.6g}; max {stats['maximum_delta']:.6g} {stats['unit']}. "
                "Cell statistics are not integrated RCS or area-weighted energy.")
        except ValueError as exc:
            self.summary.setText(str(exc))

    def export_summary(self):
        if self.result is None:
            return
        try:
            stats = summarize_roi(self.result,self.roi_bounds)
        except ValueError as exc:
            self.summary.setText(str(exc)); return
        path,_ = QFileDialog.getSaveFileName(self,"Export image comparison ROI","image_comparison_roi.json","JSON (*.json)")
        if path:
            payload = {"a":self.a.name,"b":self.b.name,"bounds":self.roi_bounds,
                "x_axis":self.a.x_axis,"x_unit":self.a.x_unit,"y_axis":self.a.y_axis,"y_unit":self.a.y_unit,
                "comparison":self.combo_mode.currentData(),"resampling":self.result['policy'],
                "statistics":stats,"advisories":self.result['notes'], "a_source":self.a.metadata,"b_source":self.b.metadata}
            try:
                Path(path).write_text(json.dumps(payload,indent=2,default=lambda value:np.asarray(value).tolist()),encoding="utf-8")
            except OSError as exc:
                self.summary.setText(f"Export failed: {exc}")

    def done(self,result):
        self._dispose()
        super().done(result)

    def _dispose(self):
        self.timer.stop()
        for record in (self.a,self.b):
            if record is not None:
                record.close()
        self.a = self.b = None
        self.result = None
        if getattr(self.owner,"_image_comparison_dialog",None) is self:
            self.owner._image_comparison_dialog = None

    def closeEvent(self,event):
        self._dispose()
        super().closeEvent(event)


def open_image_comparison(owner):
    dialog = getattr(owner,"_image_comparison_dialog",None)
    if dialog is None:
        if not records_from_figure(owner) and getattr(owner,"_active_plot_tab","plotting") == "plotting":
            from GRIM_Backend.plotting.modes import waterfall_mode
            waterfall_mode.render(owner)
        dialog = ImageComparisonDialog(owner)
        owner._image_comparison_dialog = dialog
    else:
        dialog.refresh_sources()
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
