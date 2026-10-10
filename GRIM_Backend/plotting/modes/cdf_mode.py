"""Cumulative distribution (or exceedance) of levels over the selection.

Each dataset/polarization contributes one curve pooled over every selected azimuth,
elevation, and frequency. Samples are converted to the display scale one by
one (so frequency-dependent dBke is exact) before ranking; a ranking is the
same in dB or linear power, so no statistic is taken across unlike scales.
"""

from __future__ import annotations

import numpy as np

from . import common
from .azimuth_rect_mode import _plan_series


def _pooled_display(self, dataset, selection) -> np.ndarray:
    az_indices, elev_indices, freq_indices, pol_indices = selection
    power = dataset.rcs_power[np.ix_(az_indices, elev_indices, freq_indices, [pol_indices[0]])]
    frequencies = np.asarray(dataset.frequencies[freq_indices], dtype=float)
    display = self._display_from_values(
        dataset, power[..., 0], frequency_value=frequencies[None, None, :]
    )
    display = np.asarray(display, dtype=float).ravel()
    return np.sort(display[np.isfinite(display)])


def render(self) -> None:
    self.last_plot_mode = "cdf"
    self._start_plot_render()
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
    self._configure_line_budget(len(plans))
    if not self._prepare_line_plot_axes("cdf", "rectilinear", reference, datasets):
        return

    controls = getattr(self, "analysis_controls", None)
    exceedance = bool(controls is not None and controls.cdf_exceedance())
    curves = []
    low = high = None
    for name, dataset, selection in plans:
        if len(curves) >= common.MAX_LINE_SERIES:
            break
        values = _pooled_display(self, dataset, selection)
        count = values.size
        if count == 0:
            skipped.append(name)
            continue
        levels, counts = np.unique(values, return_counts=True)
        # Tied observations jump together. Inclusive P(X >= v) is left-
        # continuous: between v[i] and v[i+1], its height is P(X >= v[i+1]).
        # Inclusive P(X <= v) is right-continuous instead.
        cumulative = (np.cumsum(counts[::-1])[::-1] if exceedance
                      else np.cumsum(counts))
        percent = 100.0 * cumulative / count
        pol_value = dataset.polarizations[selection[3][0]]
        median = float(np.median(values))
        unit = self._display_unit([(name, dataset)])
        label = (
            f"{name} | Pol {pol_value}, {len(selection[0])} az × {len(selection[1])} el × "
            f"{len(selection[2])} freq, {count:,} samples, median {median:.4g} {unit}"
        )
        trace_key = (
            "cdf", exceedance,
            tuple(float(v) for v in dataset.azimuths[selection[0]]),
            tuple(float(v) for v in dataset.elevations[selection[1]]),
            tuple(float(v) for v in dataset.frequencies[selection[2]]),
            str(pol_value),
        )
        curves.append((dataset, levels, percent, label, trace_key))
        low = values[0] if low is None else min(low, values[0])
        high = values[-1] if high is None else max(high, values[-1])

    if not curves:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No finite levels in the selected cuts.{detail}")
        return
    pad = max(1.0e-9, 0.02 * (high - low))
    for dataset, levels, percent, label, trace_key in curves:
        # Draw the known tails too, including the single-level population.
        # Every overlaid curve uses the same finite display bounds.
        x = np.r_[low - pad, levels, high + pad]
        y = np.r_[100.0 if exceedance else 0.0,
                  percent, 0.0 if exceedance else 100.0]
        self._plot_bounded_line(
            self.plot_ax, x, y, label=label, dataset=dataset,
            trace_key=trace_key, drawstyle="steps-pre" if exceedance else "steps-post",
        )
    if len(plans) > common.MAX_LINE_SERIES:
        self._note_plot_render(
            f"Displayed {common.MAX_LINE_SERIES} datasets; select fewer to show the rest."
        )

    self.plot_ax.set_xlabel(self._display_axis_label(datasets))
    self.plot_ax.set_ylabel(
        "Samples at or above level (%)" if exceedance else "Samples at or below level (%)"
    )
    self._update_legend_visibility()
    for spin, value in (
        (self.spin_plot_xmin, low - pad), (self.spin_plot_xmax, high + pad),
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
