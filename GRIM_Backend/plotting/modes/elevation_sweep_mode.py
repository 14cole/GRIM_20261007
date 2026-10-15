from __future__ import annotations

import numpy as np

from . import common
from .. import sweep_analysis


def _series(self, reference, plan, *, p50_mode, az_bounds):
    """Yield (x, display, label, trace_key) for one dataset, one per frequency."""
    name, dataset, freq_indices, az_indices, elev_indices, pol_indices = plan
    yield from sweep_analysis.iter_series(
        name, dataset, reference, (az_indices, elev_indices, freq_indices, pol_indices),
        sweep_axis="elevation", config=sweep_analysis.owner_options(self),
        phase=self._button_checked(self.btn_phase), scale=self._plot_scale_mode(),
    )


def render(self) -> None:
    self.last_plot_mode = "elevation_sweep"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations to plot.")
        return
    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    polarization = self._overlay_polarizations()
    if polarization is None:
        return

    config = sweep_analysis.owner_options(self)
    individual = config["mode"] == "individual"
    p50_mode = az_values.size > 1
    skipped: list[str] = []
    plans = []
    peak_slice_cells = 0
    total_cells = 0
    for name, dataset, pol in ((name, ds, pol) for name, ds in datasets for pol in polarization):
        freq_indices = self._axis_selection_for_dataset(
            reference, dataset, "frequency", freq_values
        )
        az_indices = self._overlay_axis_selection(
            reference, dataset, "azimuth", az_values
        )
        elev_indices = self._overlay_axis_selection(
            reference, dataset, "elevation", elev_values
        )
        pol_indices = self._indices_for_values(
            dataset.polarizations, [pol], tol=0.0
        )
        if any(value is None for value in (freq_indices, az_indices, elev_indices, pol_indices)):
            skipped.append(f"{name} | Pol {pol}")
            continue
        slice_cells = len(az_indices) * len(elev_indices)
        peak_slice_cells = max(peak_slice_cells, slice_cells)
        total_cells += slice_cells * len(freq_indices)
        plans.append(
            (name, dataset, freq_indices, az_indices, elev_indices, pol_indices)
        )

    if not plans:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(
            "No compatible data for the selected elevation, azimuth, "
            f"frequency, and polarization values.{detail}"
        )
        return
    self._configure_line_budget(sum(len(plan[2]) * (len(plan[3]) if individual else 1) for plan in plans))
    from ..requests import PlotPlan, PlotRequest, prepare_sweep
    request = PlotRequest("elevation_sweep", reference,
        tuple(PlotPlan(name, dataset, (az, el, fr, pol)) for name, dataset, fr, az, el, pol in plans),
        phase=self._button_checked(self.btn_phase), scale=self._plot_scale_mode(),
        options=config, selectors={"azimuth": az_values, "frequency": freq_values},
        max_series=common.MAX_LINE_SERIES)

    def publish(result):
        if not self._prepare_line_plot_axes(
            "elevation_sweep", "rectilinear", reference, datasets
        ):
            return

        self.plot_figure.set_layout_engine("constrained")
        self.plot_figure._grim_analysis_constrained_layout = True

        rendered, omitted = len(result.items), result.omitted
        for plan, (x_values, display, label, trace_key) in result.items:
            line_breaks = common.selection_breaks(reference, plan.dataset, "elevation", plan.selection[1], elev_values)
            self._plot_bounded_line(self.plot_ax, x_values, display, label=label,
                dataset=plan.dataset, trace_key=trace_key, break_before=line_breaks)

        if rendered == 0:
            detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
            self._show_plot_status(
                "No compatible data for the selected elevation, azimuth, "
                f"frequency, and polarization values.{detail}"
            )
            return
        if omitted:
            self._note_plot_render(
                f"Displayed {rendered} series; {omitted} further candidate series were not evaluated. "
                "Narrow frequency selections to show the rest."
            )

        self.plot_ax.set_xlabel(self._plot_axis_label(reference, "elevation"))
        tag = sweep_analysis.axis_tag(config, phase=self._button_checked(self.btn_phase)) if p50_mode or config["mode"] != "median" else ""
        self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=tag))
        self._update_legend_visibility()
        self.spin_plot_xmin.blockSignals(True)
        self.spin_plot_xmax.blockSignals(True)
        common.set_spin_value(self.spin_plot_xmin, float(elev_values[0]))
        common.set_spin_value(self.spin_plot_xmax, float(elev_values[-1]))
        self.spin_plot_xmin.blockSignals(False)
        self.spin_plot_xmax.blockSignals(False)
        self._apply_plot_limits()
        status = "Elevation sweep plot updated."
        if skipped:
            status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
        self._show_plot_status(status)

    return self._submit_plot_preparation(request, lambda cancel: prepare_sweep(request, cancel), publish, cells=total_cells)
