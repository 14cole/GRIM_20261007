from __future__ import annotations

import numpy as np

from . import common
from .azimuth_rect_mode import _plan_series, _series


def render(self) -> None:
    self.last_plot_mode = "azimuth_polar"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations to plot.")
        return
    polarization = self._overlay_polarizations()
    if polarization is None:
        return

    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or freq_values.size > 1 or elev_values.size > 1
    )
    plans, skipped = _plan_series(self, reference, datasets, az_values, elev_values, freq_values, polarization)
    if not plans or (pbp_active and getattr(self, "_plot_selection_failed", False)):
        self._show_plot_status("No compatible one-to-one coordinates for the selected plot.")
        return
    self._configure_line_budget(sum(len(sel[1]) * len(sel[2]) for _, _, sel in plans))
    angular_unit = self._plot_axis_unit(reference, "azimuth")
    self._polar_display_unit = angular_unit
    if not self._prepare_line_plot_axes(
        "azimuth_polar",
        "polar",
        reference,
        datasets,
    ):
        return

    rendered = 0
    omitted = 0
    bands = self._new_pbp_bands(datasets) if pbp_active else None
    for name, dataset, selection in plans:
        candidates = len(selection[1]) * len(selection[2])
        if bands is None and rendered >= common.MAX_LINE_SERIES:
            omitted += candidates
            continue
        for candidate_index, (x_values, display, label, trace_key) in enumerate(_series(
            self, reference, dataset, name, selection, polarization
        )):
            if not np.any(np.isfinite(display)):
                continue
            if bands is not None:
                bands.update(dataset, display, polarization=(
                    dataset.polarizations[selection[3][0]]
                ))
                rendered += 1
            elif rendered < common.MAX_LINE_SERIES:
                theta = common.convert_axis_values(
                    x_values, "azimuth", angular_unit, "rad"
                )
                self._plot_bounded_line(self.plot_ax, theta, display, label=label,
                                        dataset=dataset, trace_key=trace_key)
                rendered += 1
                if rendered >= common.MAX_LINE_SERIES:
                    omitted += candidates - candidate_index - 1
                    break

    if bands is not None:
        freq_unit = self._plot_axis_unit(reference, "frequency")
        elev_unit = self._plot_axis_unit(reference, "elevation")
        elev_name = self._plot_axis_name(reference, "elevation")
        freq_label = (
            f"{freq_values[0]:g}-{freq_values[-1]:g} {freq_unit}"
            if freq_values.size > 1
            else f"{freq_values[0]:g} {freq_unit}"
        )
        elev_label = (
            f"{elev_values[0]:g}-{elev_values[-1]:g} {elev_unit}"
            if elev_values.size > 1
            else f"{elev_values[0]:g} {elev_unit}"
        )
        pol_label = f"Pol {polarization[0]}, " if len(polarization) == 1 else ""
        bands.draw(
            az_values, f"{pol_label}Freq {freq_label}, {elev_name} {elev_label}",
            polar=True,
            to_plot_x=lambda x: common.convert_axis_values(x, "azimuth", angular_unit, "rad"),
        )

    if rendered == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(
            "No compatible data for the selected azimuth, elevation, "
            f"frequency, and polarization values.{detail}"
        )
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {rendered} series; {omitted} further candidate series were not evaluated. "
            "Narrow frequency/elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets))
    self._update_legend_visibility()

    half_turn = float(
        common.convert_axis_values([180.0], "azimuth", "deg", angular_unit)[0]
    )
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    common.set_spin_value(self.spin_plot_xmin, -half_turn)
    common.set_spin_value(self.spin_plot_xmax, half_turn)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)

    self._apply_plot_limits()
    status = "Azimuth/Aspect (Polar) plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
