from __future__ import annotations

import numpy as np

from . import common


def _indices(self, reference, dataset, azimuths, elevations, frequencies, polarization):
    az_indices = self._overlay_axis_selection(reference, dataset, "azimuth", azimuths)
    elev_indices = self._axis_selection_for_dataset(reference, dataset, "elevation", elevations)
    freq_indices = self._axis_selection_for_dataset(reference, dataset, "frequency", frequencies)
    pol_indices = self._indices_for_values(dataset.polarizations, [polarization], tol=0.0)
    if any(value is None for value in (az_indices, elev_indices, freq_indices, pol_indices)):
        return None
    return az_indices, elev_indices, freq_indices, pol_indices


def _series(self, reference, dataset, name, selection, polarization):
    az_indices, elev_indices, freq_indices, pol_indices = selection
    az_values = self._plot_axis_values(
        reference, dataset, "azimuth", dataset.azimuths[az_indices]
    )
    pol_value = dataset.polarizations[pol_indices[0]]
    freq_unit = self._plot_axis_unit(reference, "frequency")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    elev_name = self._plot_axis_name(reference, "elevation")
    selected_azimuths = tuple(float(value) for value in dataset.azimuths[az_indices])
    for freq_idx in freq_indices:
        native_frequency = float(dataset.frequencies[freq_idx])
        frequency = float(
            self._plot_axis_values(reference, dataset, "frequency", [native_frequency])[0]
        )
        for elev_idx in elev_indices:
            native_elevation = float(dataset.elevations[elev_idx])
            elevation = float(
                self._plot_axis_values(reference, dataset, "elevation", [native_elevation])[0]
            )
            if self._button_checked(self.btn_phase):
                raw = dataset.rcs_slice((az_indices, elev_idx, freq_idx, pol_indices[0]))
            else:
                raw = dataset.rcs_power[az_indices, elev_idx, freq_idx, pol_indices[0]]
            display = self._display_from_values(
                dataset, raw, frequency_value=native_frequency
            )
            label = (
                f"{name} | Pol {pol_value}, Freq {frequency:.12g} {freq_unit}, "
                f"{elev_name} {elevation:.12g} {elev_unit}"
            )
            trace_key = (selected_azimuths, native_elevation, native_frequency, str(pol_value))
            yield az_values, np.asarray(display), label, trace_key


def _plan_series(self, reference, datasets, az_values, elev_values, freq_values, polarization):
    plans, skipped = [], []
    polarizations = (polarization,) if isinstance(polarization, str) else polarization
    for name, dataset in datasets:
        for pol in polarizations:
            selection = _indices(self, reference, dataset, az_values, elev_values, freq_values, pol)
            if selection is None:
                skipped.append(f"{name} | Pol {pol}")
            else:
                plans.append((name, dataset, selection))
    return plans, skipped


def render(self) -> None:
    self.last_plot_mode = "azimuth_rect"
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
    if not self._prepare_line_plot_axes(
        "azimuth_rect",
        "rectilinear",
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
        series = _series(self, reference, dataset, name, selection, polarization)
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
            polar=False,
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
    self._apply_plot_limits()
    status = "Azimuth/Aspect (Rect) plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
