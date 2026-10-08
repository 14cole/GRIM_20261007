from __future__ import annotations

import warnings

import numpy as np

from . import common


def _frequency_selection(
    self, reference, dataset, freq_values, az_values, elev_values, polarization
):
    freq_indices = self._overlay_axis_selection(
        reference, dataset, "frequency", freq_values
    )
    az_indices = self._overlay_axis_selection(
        reference, dataset, "azimuth", az_values
    )
    elev_indices = self._axis_selection_for_dataset(
        reference, dataset, "elevation", elev_values
    )
    pol_indices = self._indices_for_values(dataset.polarizations, [polarization], tol=0.0)
    if any(value is None for value in (freq_indices, az_indices, elev_indices, pol_indices)):
        return None
    return freq_indices, az_indices, elev_indices, pol_indices


def _frequency_series(
    self,
    reference,
    dataset,
    name,
    freq_values,
    az_values,
    elev_values,
    polarization,
    *,
    selection=None,
):
    if selection is None:
        selection = _frequency_selection(
            self,
            reference,
            dataset,
            freq_values,
            az_values,
            elev_values,
            polarization,
        )
    if selection is None:
        return None
    freq_indices, az_indices, elev_indices, pol_indices = selection

    native_frequencies = np.asarray(dataset.frequencies[freq_indices], dtype=float)
    display_frequencies = self._plot_axis_values(
        reference, dataset, "frequency", native_frequencies
    )
    pol_value = dataset.polarizations[pol_indices[0]]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    az_name = self._plot_axis_name(reference, "azimuth")
    az_unit = self._plot_axis_unit(reference, "azimuth")
    az_min, az_max = float(np.min(az_values)), float(np.max(az_values))
    selected_azimuths = tuple(float(value) for value in dataset.azimuths[az_indices])
    selected_frequencies = tuple(float(value) for value in native_frequencies)

    def iter_series():
        for elev_idx in elev_indices:
            native_elevation = float(dataset.elevations[elev_idx])
            elevation = float(
                self._plot_axis_values(
                    reference, dataset, "elevation", [native_elevation]
                )[0]
            )
            if self._button_checked(self.btn_phase):
                raw = dataset.rcs_slice(
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                )[:, 0, :, 0]
                phase_degrees = self._phase_display_degrees(dataset, raw)
                display = self._wrap_phase_degrees(
                    dataset, self._phase_p50(phase_degrees, axis=0)
                )
            else:
                power = dataset.rcs_power[
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                ][:, 0, :, 0]
                power = np.where(np.isfinite(power), power, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p50_linear = np.nanmedian(power, axis=0)
                display = self._display_from_linear(
                    dataset, p50_linear, frequency_value=native_frequencies
                )
            label = (
                f"{name} | Pol {pol_value}, {elev_name} {elevation:.12g} {elev_unit}, "
                f"P50 over {az_name} ({az_min:.12g},{az_max:.12g}) {az_unit}"
            )
            trace_key = ("p50", selected_azimuths, native_elevation, selected_frequencies, str(pol_value))
            yield display_frequencies, np.asarray(display), label, trace_key

    return iter_series()


def render(self) -> None:
    self.last_plot_mode = "frequency"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations to plot.")
        return
    polarization = self._overlay_polarizations()
    if polarization is None:
        return

    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or elev_values.size > 1 or az_values.size > 1
    )
    skipped: list[str] = []
    plans = []
    peak_slice_cells = 0
    total_cells = 0
    for name, dataset, pol in ((name, ds, pol) for name, ds in datasets for pol in polarization):
        selection = _frequency_selection(
            self,
            reference,
            dataset,
            freq_values,
            az_values,
            elev_values,
            pol,
        )
        if selection is None:
            skipped.append(f"{name} | Pol {pol}")
            continue
        freq_indices, az_indices, elev_indices, _pol_indices = selection
        slice_cells = len(az_indices) * len(freq_indices)
        peak_slice_cells = max(peak_slice_cells, slice_cells)
        total_cells += slice_cells * len(elev_indices)
        plans.append((name, dataset, selection))

    if not plans or (pbp_active and getattr(self, "_plot_selection_failed", False)):
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(
            "No compatible data for the selected frequency, azimuth, "
            f"elevation, and polarization values.{detail}"
        )
        return
    self._configure_line_budget(sum(len(sel[2]) for _, _, sel in plans))
    try:
        common.validate_synchronous_plot_workload(
            operation="Frequency P50 plot",
            peak_slice_cells=peak_slice_cells,
            total_cells=total_cells,
        )
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return
    def series_for(plan):
        series = _frequency_series(
            self,
            reference,
            plan[1],
            plan[0],
            freq_values,
            az_values,
            elev_values,
            polarization,
            selection=plan[2],
        )
        assert series is not None
        return series

    if not self._prepare_line_plot_axes(
        "frequency",
        "rectilinear",
        reference,
        datasets,
    ):
        return

    rendered = 0
    omitted = 0
    bands = self._new_pbp_bands(datasets) if pbp_active else None
    for plan in plans:
        name, dataset, selection = plan
        candidates = len(selection[2])
        if bands is None and rendered >= common.MAX_LINE_SERIES:
            omitted += candidates
            continue
        series = series_for(plan)
        for candidate_index, (x_values, display, label, trace_key) in enumerate(series):
            if not np.any(np.isfinite(display)):
                continue
            if bands is not None:
                bands.update(dataset, display, polarization=(
                    dataset.polarizations[selection[3][0]]
                ))
                rendered += 1
            elif rendered < common.MAX_LINE_SERIES:
                self._plot_bounded_line(self.plot_ax, x_values, display, label=label,
                                        dataset=dataset, trace_key=trace_key)
                rendered += 1
                if rendered >= common.MAX_LINE_SERIES:
                    omitted += candidates - candidate_index - 1
                    break

    if bands is not None:
        elev_name = self._plot_axis_name(reference, "elevation")
        elev_unit = self._plot_axis_unit(reference, "elevation")
        az_name = self._plot_axis_name(reference, "azimuth")
        az_unit = self._plot_axis_unit(reference, "azimuth")
        elev_label = (
            f"{elev_values[0]:g}-{elev_values[-1]:g} {elev_unit}"
            if elev_values.size > 1
            else f"{elev_values[0]:g} {elev_unit}"
        )
        pol_label = f"Pol {polarization[0]}, " if len(polarization) == 1 else ""
        bands.draw(
            freq_values,
            f"{pol_label}{elev_name} {elev_label}, "
            f"P50 over {az_name} ({az_values[0]:g},{az_values[-1]:g}) {az_unit}",
            polar=False,
        )

    if rendered == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(
            "No compatible data for the selected frequency, azimuth, "
            f"elevation, and polarization values.{detail}"
        )
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {rendered} series; {omitted} further candidate series were not evaluated. "
            "Narrow elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "frequency"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=" P50"))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    common.set_spin_value(self.spin_plot_xmin, float(freq_values[0]))
    common.set_spin_value(self.spin_plot_xmax, float(freq_values[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    status = "Frequency plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
