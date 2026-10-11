"""Cumulative distribution (or exceedance) of levels over the selection.

Each dataset/polarization is kept separate. Optional band/sector groups and
explicit coordinate-hull quadrature define each empirical population. Samples are converted to the display scale one by
one before ranking, including frequency-dependent dBke conversion. Valid zero
power stays in the population as -infinity on a logarithmic scale. For a
frequency-dependent conversion the ranking can change; the population cannot.
"""

from __future__ import annotations

import numpy as np

from . import common
from .azimuth_rect_mode import _plan_series
from .. import distribution_analysis as distribution


def _pooled_population(self, dataset, selection):
    pop = distribution.population(dataset, selection, scale="linear" if self._plot_scale_is_linear() else "dbsm")
    return pop["values"], pop["counts"]


def _pooled_display(self, dataset, selection) -> np.ndarray:
    return _pooled_population(self, dataset, selection)[0]


def render(self) -> None:
    self.last_plot_mode = "cdf"
    self._start_plot_render()
    controls = getattr(self, "analysis_controls", None)
    if callable(getattr(controls, "set_cdf_results", None)):
        controls.set_cdf_results("")
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    if self._button_checked(self.btn_phase):
        self.status.showMessage("CDF plots rank magnitude levels. Turn off Phase to plot a CDF.")
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

    plans, skipped = _plan_series(
        self, reference, datasets, az_values, elev_values, freq_values, polarization
    )
    if not plans:
        self._show_plot_status("No compatible one-to-one coordinates for the selected plot.")
        return
    cells = [len(sel[0]) * len(sel[1]) * len(sel[2]) for _name, _dataset, sel in plans]
    try:
        common.validate_synchronous_plot_workload(
            operation="CDF", peak_slice_cells=max(cells), total_cells=sum(cells)
        )
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return
    self._configure_line_budget(common.MAX_LINE_SERIES)
    if not self._prepare_line_plot_axes("cdf", "rectilinear", reference, datasets):
        return

    controls = getattr(self, "analysis_controls", None)
    config = distribution.owner_options(self)
    exceedance = config["exceedance"]
    curves = []
    omitted = 0
    scale = "linear" if self._plot_scale_is_linear() else "dbsm"
    for name, dataset, selection in plans:
        for group_selection, group_label in distribution.groups(
            dataset, reference, selection, config, azimuths=az_values, frequencies=freq_values
        ):
            if len(curves) >= common.MAX_LINE_SERIES:
                omitted += 1
                continue
            curve = distribution.prepare_curve(name, dataset, group_selection, group_label, config, scale=scale)
            if curve is None:
                skipped.append(f"{name} | {group_label}")
            else:
                curves.append(curve)
    if not curves:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No finite levels in the selected cuts.{detail}")
        return
    self.plot_figure.set_layout_engine("constrained")
    self.plot_figure._grim_analysis_constrained_layout = True
    low, high = distribution.bounds(curves, config["threshold"] if config["threshold_enabled"] else None)
    zero_notes = []
    for curve in curves:
        x, y = distribution.draw_values(curve, (low, high), exceedance=exceedance)
        lines = self._plot_bounded_line(
            self.plot_ax, x, y, label=curve["label"], dataset=curve["dataset"],
            trace_key=curve["key"], drawstyle="steps-pre" if exceedance else "steps-post",
        )
        for line in lines:
            line._grim_cdf_population = curve["population"]["counts"]
            line._grim_cdf_weight_coverage = curve["population"]["coverage"]
            line._grim_cdf_threshold_lookup = curve["lookup"]
        if np.any(np.isneginf(curve["levels"])):
            population = curve["population"]
            zero_percent = 100. * population["zero_weight"] / population["coverage"]
            zero_notes.append(f"{population['counts']['zero']:,} zero samples ({zero_percent:g}% mass) at -∞ dB")
    if zero_notes:
        self._note_plot_render("Zero power retained in the probability denominator at -infinity dB.")
        self.plot_ax.text(0.01, 0.02, "Zero power retained: " + "; ".join(zero_notes),
                          transform=self.plot_ax.transAxes, fontsize=8,
                          color=self._current_plot_text(), va="bottom",
                          bbox={"facecolor": self._current_plot_bg(), "alpha": 0.85, "edgecolor": "none"})
    if omitted:
        self._note_plot_render(f"Displayed {len(curves)} groups; {omitted} further groups were not evaluated. Narrow the selection to show the rest.")
    if config["weighting"] != "samples":
        self._note_plot_render("CDF uses normalized trapezoidal weights over each selected group coordinate hull, including gaps; not dwell-time or solid angle.")
    if config["threshold_enabled"]:
        self.plot_ax.axvline(config["threshold"], color=self._current_plot_text(), linestyle=":", label="_CDF threshold")
        low, high = min(low, config["threshold"]), max(high, config["threshold"])
        self._note_plot_render("Inclusive threshold lookup and coverage are available in CDF Population / threshold.")
    if callable(getattr(controls, "set_cdf_results", None)):
        text = distribution.table_text(curves, config)
        if omitted:
            text += f"\n{omitted} groups omitted by the visible-series limit."
        controls.set_cdf_results(text)
    self.plot_ax.set_xlabel(self._display_axis_label(datasets))
    self.plot_ax.set_ylabel(distribution.y_label(config))
    self._update_legend_visibility()
    for spin, value in (
        (self.spin_plot_xmin, low), (self.spin_plot_xmax, high),
        (self.spin_plot_ymin, 0.0), (self.spin_plot_ymax, 100.0),
    ):
        spin.blockSignals(True)
        common.set_spin_value(spin, float(value))
        spin.blockSignals(False)
    self._apply_plot_limits()
    status = "Exceedance plot updated." if exceedance else "CDF plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
