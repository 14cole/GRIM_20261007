"""Headless image figure ownership for exact temporary native samples."""
from matplotlib.figure import Figure

from .requests import PlotResult


class WaterfallFigure(Figure):
    """Release mapped native samples when a script clears or closes a figure."""

    def _release_waterfall(self, event=None):
        for callback in getattr(self, "_grim_heatmap_cut_callbacks", []):
            self.canvas.mpl_disconnect(callback)
        self._grim_heatmap_cut_callbacks = []
        PlotResult(None, getattr(self, "_grim_waterfall_panels", [])).close()
        self._grim_waterfall_panels = []

    def clear(self, keep_observers=False):
        self._release_waterfall()
        super().clear(keep_observers=keep_observers)
