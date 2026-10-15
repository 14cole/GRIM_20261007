"""Public plot identities and replay capabilities, without GUI imports.

Keep capability decisions shared between the recorder and Python renderer.
Button placement and presentation belong to their respective workspaces.
"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PlotMode:
    key: str
    label: str
    headless: bool = False


PLOT_MODES = MappingProxyType({mode.key: mode for mode in (
    PlotMode("azimuth_rect", "Azimuth", True),
    PlotMode("azimuth_polar", "Polar", True),
    PlotMode("frequency", "Frequency", True),
    PlotMode("elevation_sweep", "Elevation Sweep", True),
    PlotMode("cdf", "CDF", True),
    PlotMode("waterfall", "Waterfall", True),
    PlotMode("isar_image", "ISAR", True),
    PlotMode("delta_map", "Delta Map", True),
    PlotMode("compare", "Compare"),
    PlotMode("sector_stats", "Sector Statistics"),
    PlotMode("range_freq", "Range / Frequency"),
    PlotMode("az_vs_range", "Azimuth / Downrange"),
)})
HEADLESS_PLOT_MODES = frozenset(key for key, mode in PLOT_MODES.items() if mode.headless)


def canonical_plot_mode(mode):
    key = str(mode).strip().lower()
    return "waterfall" if key == "heatmap" else key
