"""Azimuth sector statistics drawn as level segments over each sector.

Each dataset/frequency/elevation cut becomes one line of flat segments, one
per sector, on the same azimuth axis as Azimuth (Rect). Statistics use linear
power (a mean of dB values is not a mean level), including valid zero-power
samples. Every statistic is kept for Copy Sector Table; logarithmic zero
is represented by negative infinity, separately from missing data (NaN).
"""

from __future__ import annotations

import warnings

import numpy as np

from . import common
from .azimuth_rect_mode import _plan_series

STATISTICS = (
    ("mean", "Mean"),
    ("median", "Median"),
    ("max", "Max"),
    ("min", "Min"),
    ("percentile", "Percentile"),
)


def _sector_linear_statistics(linear, members, percentile: float) -> dict[str, float]:
    values = linear[members]
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "max": np.nan,
                "min": np.nan, "percentile": np.nan}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "percentile": float(np.percentile(values, percentile)),
    }


def statistic_tag(statistic: str, percentile: float) -> str:
    return f"P{percentile:g}" if statistic == "percentile" else statistic


def render(self) -> None:
    self.last_plot_mode = "sector_stats"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    if self._button_checked(self.btn_phase):
        self.status.showMessage(
            "Sector statistics summarise magnitude levels. Turn off Phase to plot them."
        )
        return

    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    for values, axis in ((az_values, "azimuths"), (freq_values, "frequencies"),
                         (elev_values, "elevations")):
        if values.size == 0:
            self.status.showMessage(f"Select one or more {axis} to plot.")
            return
    polarization = self._overlay_polarizations()
    if polarization is None:
        return

    controls = getattr(self, "analysis_controls", None)
    statistic = controls.sector_statistic() if controls is not None else "mean"
    percentile = controls.sector_percentile() if controls is not None else 90.0
    period = float(common.convert_axis_values(
        [360.0], "azimuth", "deg", self._plot_axis_unit(reference, "azimuth")
    )[0])
    try:
        sector_text = (
            controls.sector_text(unit=self._plot_axis_unit(reference, "azimuth"))
            if controls is not None else "30"
        )
        sectors = common.parse_sectors(sector_text, az_values, period=period)
    except ValueError as exc:
        self.status.showMessage(f"Sector Stats blocked: {exc}.")
        return

    plans, skipped = _plan_series(
        self, reference, datasets, az_values, elev_values, freq_values, polarization
    )
    if not plans:
        self._show_plot_status("No compatible one-to-one coordinates for the selected plot.")
        return
    self._configure_line_budget(sum(len(sel[1]) * len(sel[2]) for _, _, sel in plans))
    if not self._prepare_line_plot_axes("sector_stats", "rectilinear", reference, datasets):
        return
    previous_zero_note = getattr(self.plot_ax, "_grim_sector_zero_note", None)
    if previous_zero_note is not None:
        if previous_zero_note in self.plot_ax.texts:
            previous_zero_note.remove()
        self.plot_ax._grim_sector_zero_note = None

    tag = statistic_tag(statistic, percentile)
    freq_unit = self._plot_axis_unit(reference, "frequency")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    elev_name = self._plot_axis_name(reference, "elevation")
    unit = self._display_unit(datasets)
    low, high = float(az_values[0]), float(az_values[-1])
    table = []
    row_units = []
    rendered = 0
    omitted = 0
    zero_levels = 0
    for name, dataset, selection in plans:
        native_unit = self._display_unit([(name, dataset)])
        az_indices, elev_indices, freq_indices, pol_indices = selection
        azimuths = self._plot_axis_values(reference, dataset, "azimuth", dataset.azimuths[az_indices])
        memberships = [sector.contains(azimuths) for sector in sectors]
        pol_value = dataset.polarizations[pol_indices[0]]
        for freq_idx in freq_indices:
            native_frequency = float(dataset.frequencies[freq_idx])
            frequency = float(self._plot_axis_values(
                reference, dataset, "frequency", [native_frequency])[0])
            for elev_idx in elev_indices:
                if rendered >= common.MAX_LINE_SERIES:
                    omitted += 1
                    continue
                elevation = float(self._plot_axis_values(
                    reference, dataset, "elevation", [float(dataset.elevations[elev_idx])])[0])
                # Stored rcs_power is already linear. rcs_to_linear clamps
                # negative real inputs to zero, which would make invalid
                # samples indistinguishable from valid physical zeros here.
                linear = np.asarray(
                    dataset.rcs_power[az_indices, elev_idx, freq_idx, pol_indices[0]],
                    dtype=float,
                )
                x_points, y_points = [], []
                for sector, members in zip(sectors, memberships):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        stats = _sector_linear_statistics(linear, members, percentile)
                    keys = ("mean", "median", "max", "min", "percentile")
                    values = np.asarray([stats[key] for key in keys])
                    if not self._plot_scale_is_linear():
                        # The shared line-display helper omits zero. Tables
                        # must instead distinguish log(0)=-inf from missing
                        # statistics, and retain the complete population.
                        values = dataset.linear_to_default_db(
                            values, frequency_value=native_frequency, eps=0.0
                        )
                        if any(stats[key] == 0.0 for key in keys):
                            self._note_plot_render(
                                "Zero-power statistics are retained as -inf in the sector table."
                            )
                    shown = {key: float(value) for key, value in zip(keys, values)}
                    table.append((
                        name, str(pol_value), frequency, elevation, sector.label(),
                        stats["count"], shown["mean"], shown["median"], shown["min"],
                        shown["max"], shown["percentile"],
                    ))
                    row_units.append(native_unit)
                    level = shown[statistic]
                    if np.isneginf(level):
                        zero_levels += 1
                    if not np.isfinite(level):
                        continue
                    for first, last in sector.display_pieces(low, high):
                        x_points += [first, last, np.nan]
                        y_points += [level, level, np.nan]
                if not x_points:
                    continue
                label = (
                    f"{name} | Pol {pol_value}, Freq {frequency:.12g} {freq_unit}, "
                    f"{elev_name} {elevation:.12g} {elev_unit}, sector {tag}"
                )
                trace_key = (
                    "sector", statistic, percentile, tuple((s.start, s.width) for s in sectors),
                    tuple(float(v) for v in dataset.azimuths[az_indices]),
                    native_frequency, float(dataset.elevations[elev_idx]), str(pol_value),
                )
                self._plot_bounded_line(
                    self.plot_ax, np.asarray(x_points), np.asarray(y_points),
                    label=label, dataset=dataset, trace_key=trace_key,
                    linewidth=2.5, solid_capstyle="butt",
                )
                rendered += 1

    self.plot_figure._grim_sector_table = {
        "unit": unit,
        "percentile": percentile,
        "rows": table,
        "row_units": row_units,
    }
    if zero_levels:
        zero_message = (
            f"{zero_levels} zero-power sector level(s) (-inf on the dB scale). "
            "Use Linear to display these levels."
        )
        self._note_plot_render(zero_message)
        # Include the explanation in exported figures as well as the status
        # bar: there is no finite ordinate at which to draw these levels.
        self.plot_ax._grim_sector_zero_note = self.plot_ax.text(
            0.02, 0.02,
            f"{zero_levels} zero-power sector level(s): -inf on the dB scale\n"
            "Retained in the sector table; use Linear to display.",
            transform=self.plot_ax.transAxes, ha="left", va="bottom", fontsize="small",
        )
    if rendered == 0 and zero_levels == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No finite levels inside the sectors.{detail}")
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {rendered} series; {omitted} further cuts were not drawn. "
            "Narrow frequency/elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=f" {tag}"))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    common.set_spin_value(self.spin_plot_xmin, low)
    common.set_spin_value(self.spin_plot_xmax, high)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    status = f"Sector statistics ({len(sectors)} sectors, {tag}) plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)


def table_text(table) -> str:
    """Tab-separated sector table for the clipboard."""
    unit = table["unit"]
    row_units = table.get("row_units", [])
    mixed_units = len(set(row_units)) > 1
    suffix = "" if mixed_units else f" ({unit})"
    header = (
        "Dataset", "Pol", "Frequency", "Elevation", "Sector", "Samples",
        f"Mean{suffix}", f"Median{suffix}", f"Min{suffix}", f"Max{suffix}",
        f"P{table['percentile']:g}{suffix}",
    )
    if mixed_units:
        header += ("Unit",)
    lines = ["\t".join(header)]
    for row_index, row in enumerate(table["rows"]):
        cells = [str(value) if not isinstance(value, float) else f"{value:.6g}" for value in row]
        if mixed_units:
            cells.append(row_units[row_index])
        lines.append("\t".join(cells))
    return "\n".join(lines)
