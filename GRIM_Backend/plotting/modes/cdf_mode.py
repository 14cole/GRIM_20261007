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


def clear_state(figure):
    """Release the exact distribution retained for threshold-only edits."""
    figure._grim_cdf_state = None


def update_threshold(owner):
    """Update a threshold in O(groups log levels), without rereading samples.

    Returns False when selectors/source/population settings require a fresh
    preparation. Canvas/source invalidation calls ``clear_state``.
    """
    state = getattr(owner.plot_figure, "_grim_cdf_state", None)
    if not state or state["axes"] is not owner.plot_ax:
        return False
    if state["generation"] != getattr(owner, "_plot_render_generation", None):
        return False
    config = distribution.owner_options(owner)
    fixed = lambda values: {key: value for key, value in values.items()
                            if key not in {"threshold", "threshold_enabled"}}
    if fixed(config) != fixed(state["config"]):
        return False
    curves = state["curves"]
    low, high = distribution.bounds(curves, config["threshold"] if config["threshold_enabled"] else None)
    for curve, lines in zip(curves, state["lines"]):
        curve["lookup"] = distribution.threshold_lookup(curve, config["threshold"])
        for line in lines:
            if line.axes is not owner.plot_ax:
                return False
            line._grim_cdf_threshold_lookup = curve["lookup"]
            # Only artificial constant-probability tails move. Measured levels
            # and every retained exact probability remain unchanged.
            x = np.array(line.get_xdata(), copy=True)
            x[0], x[-1] = low, high
            line.set_xdata(x)
            native = getattr(line, "_grim_native_samples", None)
            if native is not None and native.values is not None:
                native.values[0, 0], native.values[0, -1] = low, high
    artist = state["threshold_artist"]
    if config["threshold_enabled"]:
        if artist is None:
            artist = owner.plot_ax.axvline(config["threshold"], color=owner._current_plot_text(), linestyle=":", label="_CDF threshold")
        else:
            artist.set_xdata([config["threshold"], config["threshold"]])
            artist.set_visible(True)
    elif artist is not None:
        artist.set_visible(False)
    controls = getattr(owner, "analysis_controls", None)
    if callable(getattr(controls, "set_cdf_results", None)):
        text = distribution.table_text(curves, config)
        if state["omitted"]:
            text += f"\n{state['omitted']} groups omitted by the visible-series limit."
        controls.set_cdf_results(text)
    for spin, value in ((owner.spin_plot_xmin, low), (owner.spin_plot_xmax, high)):
        spin.blockSignals(True)
        common.set_spin_value(spin, value)
        spin.blockSignals(False)
    owner._apply_plot_limits()
    owner.plot_figure.canvas.draw_idle()
    state.update(config=config, threshold_artist=artist)
    owner._show_plot_status("CDF threshold updated from the exact retained distribution.")
    return True


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
    from ..requests import PlotPlan, PlotRequest, prepare_cdf
    config = distribution.owner_options(self)
    exceedance = config["exceedance"]
    request = PlotRequest("cdf", reference,
        tuple(PlotPlan(name, dataset, selection) for name, dataset, selection in plans),
        scale="linear" if self._plot_scale_is_linear() else "dbsm", options=config,
        selectors={"azimuth": az_values, "frequency": freq_values}, max_series=common.MAX_LINE_SERIES)

    def publish(result):
        self._configure_line_budget(common.MAX_LINE_SERIES)
        if not self._prepare_line_plot_axes("cdf", "rectilinear", reference, datasets):
            return
        curves, omitted = result.items, result.omitted
        skipped.extend(result.skipped)
        if not curves:
            detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
            self._show_plot_status(f"No finite levels in the selected cuts.{detail}")
            return
        self.plot_figure.set_layout_engine("constrained")
        self.plot_figure._grim_analysis_constrained_layout = True
        low, high = distribution.bounds(curves, config["threshold"] if config["threshold_enabled"] else None)
        zero_notes = []
        curve_lines = []
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
            curve_lines.append(lines)
            if len(curve["levels"]) and np.isneginf(curve["levels"][0]):
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
        threshold_artist = None
        if config["threshold_enabled"]:
            threshold_artist = self.plot_ax.axvline(config["threshold"], color=self._current_plot_text(), linestyle=":", label="_CDF threshold")
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
        self.plot_figure._grim_cdf_state = dict(curves=curves, config=dict(config), lines=curve_lines,
            axes=self.plot_ax, generation=getattr(self, "_plot_render_generation", None),
            threshold_artist=threshold_artist, omitted=omitted)
        status = "Exceedance plot updated." if exceedance else "CDF plot updated."
        if skipped:
            status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
        self._show_plot_status(status)

    return self._submit_plot_preparation(request, lambda cancel: prepare_cdf(request, cancel), publish, cells=sum(cells))
