from __future__ import annotations

import math
from typing import Mapping


def style_axis(axis: object, colors: Mapping[str, str]) -> None:
    axis.set_facecolor(colors["plot_axes_bg"])
    axis.xaxis.label.set_color(colors["plot_text"])
    axis.yaxis.label.set_color(colors["plot_text"])
    axis.title.set_color(colors["plot_text"])
    axis.tick_params(axis="x", colors=colors["plot_text"])
    axis.tick_params(axis="y", colors=colors["plot_text"])
    for spine in axis.spines.values():
        spine.set_color(colors["plot_spine"])


def style_colorbar(colorbar: object, colors: Mapping[str, str]) -> None:
    colorbar.ax.tick_params(colors=colors["plot_text"])
    colorbar.ax.yaxis.label.set_color(colors["plot_text"])
    colorbar.ax.set_facecolor(colors["plot_axes_bg"])
    colorbar.outline.set_edgecolor(colors["plot_spine"])


def format_readout(value: float) -> str:
    """Plotted value in hover and selection callouts (comparison-table precision)."""
    return f"{float(value):.5g}"


def data_range(*grids: object) -> tuple[float, float] | None:
    """Finite (min, max) over the grids: matplotlib's automatic color limits."""
    import numpy as np

    values = np.concatenate([np.asarray(grid, dtype=float).ravel() for grid in grids])
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else None


def fixed_color_limits(low: object, high: object) -> tuple[float, float]:
    """Validated user color-map limits, from text fields or a project file."""
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError):
        raise ValueError("Min and Max must be numbers.") from None
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("Min and Max must be finite.")
    if high <= low:
        raise ValueError("Max must be greater than Min.")
    return low, high


def show_grid_value_on_hover(axis: object, x: object, y: object, values: object, *, edges: bool = False) -> None:
    """Add the map value under the cursor to the toolbar's (x, y) hover callout.

    Matplotlib disables cursor data for pcolormesh, so its toolbar would show
    only x and y. ``values`` is indexed [y][x], as passed to pcolormesh; ``x``
    and ``y`` are sample centers (``shading='nearest'``) or, with ``edges``,
    the cell edges. The value is the drawn cell's sample: no interpolation.
    """
    import numpy as np

    def cell_edges(centers):
        c = np.asarray(centers, dtype=float)
        if len(c) < 2:
            return np.repeat(c, 2)  # matplotlib's zero-width 'nearest' cell
        middle = (c[:-1] + c[1:]) / 2
        return np.concatenate(([2 * c[0] - middle[0]], middle, [2 * c[-1] - middle[-1]]))

    x_edges = np.asarray(x, dtype=float) if edges else cell_edges(x)
    y_edges = np.asarray(y, dtype=float) if edges else cell_edges(y)
    if np.shape(values) != (len(y_edges) - 1, len(x_edges) - 1):
        raise ValueError(f"Map values {np.shape(values)} do not match {len(y_edges) - 1} x {len(x_edges) - 1} cells.")

    def cell(bounds, value):
        count = len(bounds) - 1
        ascending = bounds[-1] >= bounds[0]
        ordered = bounds if ascending else bounds[::-1]
        if not ordered[0] <= value <= ordered[-1]:  # also rejects NaN
            return None
        index = min(int(np.searchsorted(ordered, value, side="right")) - 1, count - 1)
        return index if ascending else count - 1 - index

    def format_coord(x_value, y_value):
        column = None if x_value is None else cell(x_edges, x_value)
        row = None if y_value is None else cell(y_edges, y_value)
        value = None if column is None or row is None else values[row][column]
        if value is None or np.ma.is_masked(value):
            return type(axis).format_coord(axis, x_value, y_value)  # matplotlib's (x, y)
        # One line: the toolbar is a single row and a second line would clip.
        return f"(x, y, z) = ({axis.format_xdata(x_value)}, {axis.format_ydata(y_value)}, {format_readout(value)})"

    axis.format_coord = format_coord
