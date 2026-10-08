"""Scrub one parameter axis and re-plot at each value.

The slider selects exactly one value in the chosen parameter list and
re-renders the current plot type, independent of Auto Plot. Under Hold the
curves and bands the slider added last time are replaced, so a scrubbed plot
can move over held PBP bands or reference curves without piling up.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer

# The axis each plot type sweeps along; the slider must scrub a different one.
SWEEP_AXIS = {
    "azimuth_rect": "azimuth",
    "azimuth_polar": "azimuth",
    "sector_stats": "azimuth",
    "frequency": "frequency",
    "elevation_sweep": "elevation",
    "range_freq": "frequency",
}


class PlotSliderMixin:
    def _slider_list(self, axis: str):
        return {
            "frequency": self.list_freq,
            "elevation": self.list_elev,
            "azimuth": self.list_az,
        }[axis]

    def _on_plot_slider_toggled(self, checked: bool) -> None:
        bar = getattr(self, "plot_slider", None)
        if bar is None:
            return
        bar.setVisible(bool(checked))
        if checked:
            self._refresh_plot_slider()
        else:
            bar.stop_play()

    def _schedule_plot_slider_refresh(self, *_args) -> None:
        """Coalesce list rebuilds (one signal per inserted row) into one refresh."""
        timer = getattr(self, "_plot_slider_refresh_timer", None)
        if timer is None:
            timer = self._plot_slider_refresh_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(0)
            timer.timeout.connect(self._refresh_plot_slider)
        timer.start()

    def _refresh_plot_slider(self, *_args) -> None:
        """Match the slider range and position to the chosen parameter list."""
        bar = getattr(self, "plot_slider", None)
        if bar is None or not bar.isVisible():
            return
        widget = self._slider_list(bar.axis())
        rows = [widget.row(item) for item in widget.selectedItems()]
        bar.set_positions(
            [widget.item(row).text() for row in range(widget.count())],
            min(rows) if rows else 0,
        )

    def _on_plot_slider_moved(self, row: int) -> None:
        bar = getattr(self, "plot_slider", None)
        if bar is None:
            return
        axis = bar.axis()
        widget = self._slider_list(axis)
        item = widget.item(int(row))
        if item is None:
            return
        mode = getattr(self, "last_plot_mode", None)
        if SWEEP_AXIS.get(mode) == axis:
            bar.stop_play()
            self.status.showMessage(
                f"This plot sweeps {axis}; choose another slider axis to scrub."
            )
            return
        blocked = widget.blockSignals(True)
        try:
            widget.clearSelection()
            item.setSelected(True)
            widget.scrollToItem(item)
        finally:
            widget.blockSignals(blocked)
        self._invalidate_isar_result()
        if mode is None:
            return
        hold = self._button_checked(getattr(self, "btn_hold", None))
        before = set()
        if hold:
            for ax in self.plot_figure.axes:
                for artist in (*ax.lines, *ax.collections):
                    if getattr(artist, "_grim_slider_trace", False):
                        artist.remove()
            before = {id(artist) for ax in self.plot_figure.axes
                      for artist in (*ax.lines, *ax.collections)}
        self._render_plot_mode(mode)
        if hold:
            # Keep pre-existing held bands, but replace this slider's own
            # previous curves and bands. Markers re-snap after every render.
            for ax in self.plot_figure.axes:
                for artist in (*ax.lines, *ax.collections):
                    key = getattr(artist, "_grim_dataset_key", None)
                    if id(artist) not in before and key is not None:
                        artist._grim_slider_trace = True
