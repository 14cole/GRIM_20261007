"""Heatmaps of any pair of measured RF axes; retains the waterfall recipe key."""
from __future__ import annotations

import numpy as np

from . import common


def render(self) -> None:
    from GRIM_Backend.plotting.image_analysis import AXES, AXIS_ATTRIBUTES, ImageRecord, heatmap_display_plane, compact_metadata
    self.last_plot_mode = "waterfall"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    controls = getattr(self, "image_analysis_controls", None)
    x_axis, y_axis = controls.axes() if controls is not None else ("azimuth", "frequency")
    if x_axis not in AXES or y_axis not in AXES or x_axis == y_axis:
        self.status.showMessage("Choose two different waterfall axes.")
        return
    remaining = next(axis for axis in AXES if axis not in (x_axis, y_axis))
    aggregate = controls is not None and controls.aggregation() == "mean_power"
    show_cuts = controls is not None and controls.linked_cuts()
    selected = {}
    for axis, widget in (("azimuth", self.list_az), ("elevation", self.list_elev), ("frequency", self.list_freq)):
        selected[axis] = np.asarray(sorted(self._selected_values(widget)), float)
        if not selected[axis].size:
            self.status.showMessage(f"Select one or more {axis} values to plot.")
            return
    pol_values = self._selected_values(self.list_pol)
    if not len(pol_values):
        self.status.showMessage("Select one or more polarizations to plot.")
        return
    phase = self._button_checked(self.btn_phase)
    plans, skipped = [], []
    total_source = total_display = panel_count = peak_cells = 0
    for name, dataset in datasets:
        indices = {axis: self._axis_selection_for_dataset(reference, dataset, axis, selected[axis]) for axis in AXES}
        if any(ids is None for ids in indices.values()):
            skipped.append(name)
            continue
        cells = len(indices[x_axis]) * len(indices[y_axis])
        peak_cells = max(peak_cells, cells * (len(indices[remaining]) if aggregate else 1))
        for pol in pol_values:
            pol_ids = self._indices_for_values(dataset.polarizations, [pol], tol=0.)
            if pol_ids is None:
                skipped.append(f"{name}/{pol}")
                continue
            count = 1 if aggregate else len(indices[remaining])
            panel_count += count
            total_source += cells * len(indices[remaining])
            total_display += common.bounded_image_cell_count(len(indices[x_axis]), len(indices[y_axis])) * count
            plans.append((name, dataset, indices, pol_ids))
        if phase and common.image_requires_decimation(len(indices[x_axis]), len(indices[y_axis])):
            self.status.showMessage("Phase waterfall blocked: scalar peak reduction would distort phase. Narrow the selected waterfall axes.")
            return
    if not plans:
        self._show_plot_status("No compatible selected data for this waterfall. " + ", ".join(skipped))
        return
    try:
        if panel_count > common.MAX_WATERFALL_PANELS:
            raise ValueError(f"selection creates {panel_count} panels; select fewer cuts or use mean aggregation")
        common.validate_synchronous_plot_workload(operation="Waterfall", peak_slice_cells=peak_cells, total_cells=total_source)
        common.validate_aggregate_image_cells(total_display, panel_count=panel_count, operation="Waterfall")
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return
    panels = []
    for name, dataset, indices, pol_ids in plans:
        nx = np.asarray(getattr(dataset, AXIS_ATTRIBUTES[x_axis])[indices[x_axis]])
        ny = np.asarray(getattr(dataset, AXIS_ATTRIBUTES[y_axis])[indices[y_axis]])
        x = self._plot_axis_values(reference, dataset, x_axis, nx)
        y = self._plot_axis_values(reference, dataset, y_axis, ny)
        for cut_index in ([None] if aggregate else indices[remaining]):
            native_ids = [indices[axis] for axis in AXES] + [pol_ids]
            display = heatmap_display_plane(dataset,native_ids,x_axis,y_axis,
                remaining_index=cut_index,phase=phase,linear=self._plot_scale_is_linear())
            display = np.where(np.isfinite(display), display, np.nan)
            if not np.any(np.isfinite(display)):
                continue
            bx, by, bd = self._bounded_plot_image(x, y, display)
            snapshot = self._native_image_snapshot(x, y, display, reduced=bd.shape != display.shape)
            remaining_values = getattr(dataset, AXIS_ATTRIBUTES[remaining])[indices[remaining]]
            unit = self._plot_axis_unit(reference, remaining)
            label = (f"{'Circular phase mean' if phase else 'Mean linear display quantity'} over {len(remaining_values)} {remaining} samples"
                     if aggregate else f"{remaining.capitalize()} {self._plot_axis_values(reference, dataset, remaining, [getattr(dataset, AXIS_ATTRIBUTES[remaining])[cut_index]])[0]:g} {unit}")
            title = f"{name} | Pol {dataset.polarizations[pol_ids[0]]} | {label} | {self._display_unit([(name,dataset)])}"
            short_label = f"Mean {remaining} (n={len(remaining_values)})" if aggregate else label
            panels.append(dict(name=title, short_title=f"{name[:20]} | {dataset.polarizations[pol_ids[0]]}\n{short_label} | {self._display_unit([(name,dataset)])}",dataset=dataset, x=bx, y=by, display=bd, snapshot=snapshot,
                native_x=x, native_y=y,
                metadata={"kind": "heatmap", "source_name":name, "polarization": str(dataset.polarizations[pol_ids[0]]),
                    "frequency_aggregation":bool(aggregate and remaining == "frequency"),
                    "remaining_selection": (remaining, "mean" if aggregate else float(getattr(dataset, AXIS_ATTRIBUTES[remaining])[cut_index]), common.axis_unit(dataset, remaining)),
                    "coordinate_system": dataset.units.get("angular_coordinate_system", ""),
                    "source_units": compact_metadata(dataset.units), "source_metadata": compact_metadata(dataset.extra),
                    "frequency_values_hz": np.asarray(dataset._frequency_value_to_hz(dataset.frequencies[indices['frequency']])).tolist()}))
    if not panels:
        self._show_plot_status("No finite selected samples for this waterfall.")
        return
    old_cuts = getattr(self.plot_figure, "_grim_linked_cuts", None)
    if old_cuts is not None:
        old_cuts.disconnect()
    self._remove_colorbar()
    self._release_native_plot_samples()
    self.plot_figure.clear()
    self.plot_figure.set_facecolor(self._current_plot_bg())
    self.plot_figure.set_layout_engine("constrained")
    cut_axes = []
    if show_cuts:
        # Shared cut axes compare all panels without tripling the number of
        # plot rows. Two image panels remain legible on a laptop canvas.
        outer = self.plot_figure.add_gridspec(2,1,height_ratios=[3,1.35])
        columns = min(3,len(panels))
        images = outer[0].subgridspec(int(np.ceil(len(panels)/columns)),columns)
        axes = [self.plot_figure.add_subplot(images[i//columns,i%columns]) for i in range(len(panels))]
        cuts_grid = outer[1].subgridspec(1,2,width_ratios=[3,1])
        horizontal = self.plot_figure.add_subplot(cuts_grid[0],sharex=axes[0])
        vertical = self.plot_figure.add_subplot(cuts_grid[1],sharey=axes[0])
        for ax in (horizontal,vertical):
            self._style_axes(ax)
            ax.tick_params(labelsize=7)
        cut_axes = [(horizontal,vertical)]*len(panels)
    else:
        columns = min(3,int(np.ceil(np.sqrt(len(panels)))))
        slots = self.plot_figure.subplots(int(np.ceil(len(panels)/columns)),columns,squeeze=False).ravel()
        axes = list(slots[:len(panels)])
        for extra in slots[len(panels):]:
            self.plot_figure.delaxes(extra)
    self.plot_axes, self.plot_ax = axes, axes[0]
    zmin, zmax = self.spin_plot_zmin.value(), self.spin_plot_zmax.value()
    shared = self.chk_colorbar_shared.isChecked()
    limits = common.finite_data_limits(p["display"] for p in panels) if shared else None
    vmin, vmax = (zmin, zmax) if zmin < zmax else (limits if limits else (None, None))
    meshes, records = [], []
    for panel_index,(ax, panel) in enumerate(zip(axes, panels),1):
        self._style_axes(ax)
        image = panel["display"]
        mesh = ax.pcolormesh(panel["x"], panel["y"], image.T, shading="auto", cmap=self._effective_colormap(), vmin=vmin, vmax=vmax)
        coords = mesh.get_coordinates()
        mesh._grim_rectilinear_data = (coords[0,:,0], coords[:,0,1], mesh.get_array().reshape(image.T.shape))
        self._attach_native_image_snapshot(mesh, panel["snapshot"])
        record_values = getattr(mesh, "_grim_native_rectilinear_data", None)
        if record_values is not None:
            rx, ry, rv = panel["native_x"], panel["native_y"], record_values[2]
        elif panel["snapshot"] is False:
            rx, ry, rv = panel["x"], panel["y"], mesh.get_array().reshape(image.T.shape)
        else:
            rx, ry, rv = panel["native_x"], panel["native_y"], mesh.get_array().reshape(image.T.shape)
        ds = panel["dataset"]
        record = ImageRecord(panel["name"], rx, ry, rv, x_axis, y_axis,
            self._plot_axis_unit(reference, x_axis), self._plot_axis_unit(reference, y_axis),
            "phase" if phase else ds.linear_quantity() + ("_normalized_frequency_mean" if aggregate and remaining == "frequency" and ds.default_log_unit()=="dBke" and not self._plot_scale_is_linear() else ""), self._display_unit([(panel['name'], ds)]),
            logarithmic=not phase and not self._plot_scale_is_linear(), phase=phase,
            notes=("Display block maxima; native snapshot unavailable.",) if panel["snapshot"] is False else (), metadata=panel["metadata"])
        mesh._grim_image_record = record
        records.append(record)
        meshes.append(mesh)
        compact = show_cuts or len(panels)>1
        ax.set_title((f"{panel_index} · " + panel["short_title"]) if compact else panel["name"], color=self._current_plot_text(),fontsize=9 if compact else 11)
        ax.set_xlabel(self._plot_axis_label(reference, x_axis),fontsize=8 if compact else 10,labelpad=1)
        ax.set_ylabel(self._plot_axis_label(reference, y_axis),fontsize=8 if compact else 10,labelpad=1)
        ax.tick_params(labelsize=7 if compact else 10)
    if self.chk_colorbar.isChecked():
        self.plot_colorbars = ([self.plot_figure.colorbar(meshes[-1], ax=axes)] if shared
            else [self.plot_figure.colorbar(mesh, ax=ax) for mesh, ax in zip(meshes, axes)])
        for index, colorbar in enumerate(self.plot_colorbars):
            self._apply_colorbar_ticks(colorbar)
            ds = datasets if shared else [(panels[index]["name"], panels[index]["dataset"])]
            colorbar.set_label(self._display_axis_label(ds), color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
    if show_cuts:
        from GRIM_Backend.ui.image_analysis import LinkedImageCuts
        self.plot_figure._grim_linked_cuts = LinkedImageCuts(self.plot_canvas, list(zip(axes, cut_axes, records)))
    for spin, value in ((self.spin_plot_xmin, min(np.min(p['x']) for p in panels)),
            (self.spin_plot_xmax, max(np.max(p['x']) for p in panels)),
            (self.spin_plot_ymin, min(np.min(p['y']) for p in panels)),
            (self.spin_plot_ymax, max(np.max(p['y']) for p in panels))):
        blocked = spin.blockSignals(True)
        common.set_spin_value(spin, float(value))
        spin.blockSignals(blocked)
    self._apply_plot_limits()
    self._show_plot_status("Waterfall updated." + (" Skipped: " + ", ".join(skipped) if skipped else ""))
