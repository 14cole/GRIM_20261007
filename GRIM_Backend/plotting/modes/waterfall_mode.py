"""Heatmaps of any pair of measured RF axes; retains the waterfall recipe key."""
from __future__ import annotations

import numpy as np

from . import common
from GRIM_Backend.plotting.image_grid import axis_support, image_grid, native_sample_width


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
    show_horizontal = controls is not None and getattr(controls, "horizontal_cuts", controls.linked_cuts)()
    show_vertical = controls is not None and getattr(controls, "vertical_cuts", controls.linked_cuts)()
    show_cuts = show_horizontal or show_vertical
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
        x_width = native_sample_width(reference, dataset, x_axis)
        y_width = native_sample_width(reference, dataset, y_axis)
        for cut_index in ([None] if aggregate else indices[remaining]):
            native_ids = [indices[axis] for axis in AXES] + [pol_ids]
            display = heatmap_display_plane(dataset,native_ids,x_axis,y_axis,
                remaining_index=cut_index,phase=phase,linear=self._plot_scale_is_linear())
            display = np.where(np.isfinite(display), display, np.nan)
            if not np.any(np.isfinite(display)):
                continue
            bx, by, bd = self._bounded_plot_image(x, y, display)
            snapshot = self._native_image_snapshot(x, y, display, reduced=bd.shape != display.shape)
            supported = image_grid(x, y, display.T, x_width=x_width, y_width=y_width,
                target_shape=bd.T.shape, return_samples=snapshot is False)
            x_edges, y_edges, drawn = supported[:3]
            display_samples = supported[3] if snapshot is False else None
            remaining_values = getattr(dataset, AXIS_ATTRIBUTES[remaining])[indices[remaining]]
            unit = self._plot_axis_unit(reference, remaining)
            label = (f"{'Circular phase mean' if phase else 'Mean linear display quantity'} over {len(remaining_values)} {remaining} samples"
                     if aggregate else f"{remaining.capitalize()} {self._plot_axis_values(reference, dataset, remaining, [getattr(dataset, AXIS_ATTRIBUTES[remaining])[cut_index]])[0]:g} {unit}")
            title = f"{name} | Pol {dataset.polarizations[pol_ids[0]]} | {label} | {self._display_unit([(name,dataset)])}"
            short_label = f"Mean {remaining} (n={len(remaining_values)})" if aggregate else label
            panels.append(dict(name=title, short_title=f"{name[:20]} | {dataset.polarizations[pol_ids[0]]}\n{short_label} | {self._display_unit([(name,dataset)])}",dataset=dataset, x=bx, y=by, display=bd, snapshot=snapshot,
                native_x=x, native_y=y, native_display=display if snapshot is None else None, drawn=drawn,
                x_edges=x_edges, y_edges=y_edges, x_width=x_width, y_width=y_width,
                display_samples=display_samples,
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
    horizontal = vertical = None
    if show_cuts:
        # Shared cut axes compare all panels without tripling the number of
        # plot rows. Two image panels remain legible on a laptop canvas.
        outer = (self.plot_figure.add_gridspec(2,1,height_ratios=[3,1.35]) if show_horizontal
                 else self.plot_figure.add_gridspec(1,2,width_ratios=[3,1]))
        columns = min(3,len(panels))
        images = outer[0].subgridspec(int(np.ceil(len(panels)/columns)),columns)
        axes = [self.plot_figure.add_subplot(images[i//columns,i%columns]) for i in range(len(panels))]
        if show_horizontal and show_vertical:
            cuts_grid = outer[1].subgridspec(1,2,width_ratios=[3,1])
            horizontal = self.plot_figure.add_subplot(cuts_grid[0],sharex=axes[0])
            vertical = self.plot_figure.add_subplot(cuts_grid[1],sharey=axes[0])
        elif show_horizontal:
            horizontal = self.plot_figure.add_subplot(outer[1],sharex=axes[0])
        else:
            vertical = self.plot_figure.add_subplot(outer[1],sharey=axes[0])
        for ax in (horizontal,vertical):
            if ax is not None:
                self._style_axes(ax)
                ax.tick_params(labelsize=7)
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
        image = panel["drawn"]
        mesh = ax.pcolormesh(panel["x_edges"], panel["y_edges"], image, shading="flat", cmap=self._effective_colormap(), vmin=vmin, vmax=vmax)
        mesh._grim_rectilinear_data = (panel["x_edges"], panel["y_edges"], mesh.get_array().reshape(image.shape))
        self._attach_native_image_snapshot(mesh, panel["snapshot"])
        record_values = getattr(mesh, "_grim_native_rectilinear_data", None)
        if record_values is not None:
            rx, ry, rv = panel["native_x"], panel["native_y"], record_values[2]
        elif panel["snapshot"] is False:
            samples = panel["display_samples"]
            rx, ry, rv = samples["x"], samples["y"], samples["values"]
            panel["metadata"].update({key: samples[key] for key in ("x_cut_breaks", "y_cut_breaks")})
        else:
            rx, ry, rv = panel["native_x"], panel["native_y"], panel["native_display"].T
        ds = panel["dataset"]
        record = ImageRecord(panel["name"], rx, ry, rv, x_axis, y_axis,
            self._plot_axis_unit(reference, x_axis), self._plot_axis_unit(reference, y_axis),
            "phase" if phase else ds.linear_quantity() + ("_normalized_frequency_mean" if aggregate and remaining == "frequency" and ds.default_log_unit()=="dBke" and not self._plot_scale_is_linear() else ""), self._display_unit([(panel['name'], ds)]),
            logarithmic=not phase and not self._plot_scale_is_linear(), phase=phase,
            notes=("Display block maxima; native snapshot unavailable.",) if panel["snapshot"] is False else (), metadata=panel["metadata"])
        mesh._grim_image_record = record
        if panel["snapshot"] is not False:
            # Native probes use the same local sample support as the display;
            # no midpoint cell may turn an unmeasured sector into a return.
            xs = axis_support(rx, width=panel["x_width"])
            ys = axis_support(ry, width=panel["y_width"])
            def probe(x, y, artist=mesh, xs=xs, ys=ys):
                current = getattr(artist, "_grim_image_record", None)
                ix, iy = xs.index(x), ys.index(y)
                if current is None or ix is None or iy is None:
                    return None
                value = current.values[iy, ix]
                return float(value) if np.isfinite(value) else None
            mesh._grim_native_probe = probe
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
    if show_cuts or callable(getattr(controls, "bind", None)):
        from GRIM_Backend.ui.image_analysis import LinkedImageCuts
        def can_select():
            busy = any(self._button_checked(getattr(self, role, None)) for role in ("btn_pan", "btn_zoom_box", "btn_markers"))
            overlays = getattr(self, "spatial_overlays", None)
            return not busy and not (overlays is not None and (
                overlays.drawing is not None or overlays.drag is not None))
        controller = LinkedImageCuts(self.plot_canvas,
            list(zip(axes, [(horizontal,vertical)]*len(panels), records)), can_select=can_select)
        self.plot_figure._grim_linked_cuts = controller
        if callable(getattr(controls, "bind", None)):
            controls.bind(controller)
    for spin, value in ((self.spin_plot_xmin, min(p['x_edges'][0] for p in panels)),
            (self.spin_plot_xmax, max(p['x_edges'][-1] for p in panels)),
            (self.spin_plot_ymin, min(p['y_edges'][0] for p in panels)),
            (self.spin_plot_ymax, max(p['y_edges'][-1] for p in panels))):
        blocked = spin.blockSignals(True)
        common.set_spin_value(spin, float(value))
        spin.blockSignals(blocked)
    self._apply_plot_limits()
    self._show_plot_status("Waterfall updated." + (" Skipped: " + ", ".join(skipped) if skipped else ""))
