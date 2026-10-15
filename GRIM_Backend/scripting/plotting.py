"""Render dataset selections to Matplotlib figures."""
from __future__ import annotations

from typing import Iterable, Sequence
import math
import warnings

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.requests import (
    PlotPlan, PlotRequest, prepare_sweep, prepare_cdf, prepare_waterfall_image,
)
from GRIM_Backend.plotting.registry import HEADLESS_PLOT_MODES, canonical_plot_mode


def _indices(axis, selected, *, text=False, tolerance=1.0e-6) -> list[int]:
    values = np.asarray(axis)
    if not text:
        from GRIM_Backend.plotting.modes.common import unique_axis_selection
        matched = unique_axis_selection(values, list(selected), tolerance)
        if matched is None:
            raise ValueError("Selected coordinates are missing from the dataset")
        return matched.tolist()
    result: list[int] = []
    for requested in selected:
        if text:
            matches = np.flatnonzero(values.astype(str) == str(requested))
        else:
            matches = np.flatnonzero(
                np.isclose(values.astype(float), float(requested), atol=tolerance, rtol=0.0)
            )
        if matches.size != 1:
            raise ValueError(f"Selected value {requested!r} is missing or ambiguous")
        result.append(int(matches[0]))
    return result


def _display_values(dataset: RcsGrid, values, *, frequency, phase: bool, scale: str):
    if phase:
        phase_degrees = np.rad2deg(np.angle(values))
        finite = np.isfinite(values) & (np.abs(values) > 0.)
        display = np.full(np.asarray(phase_degrees).shape, np.nan, dtype=float)
        phase_wrap = str(
            (dataset.units or {}).get("phase_wrap", "-180_180")
        ).strip()
        if phase_wrap == "0_360":
            display[finite] = np.mod(phase_degrees[finite], 360.0)
        else:
            display[finite] = (
                np.mod(phase_degrees[finite] + 180.0, 360.0) - 180.0
            )
        return display
    if str(scale).lower() == "linear":
        return dataset.rcs_to_linear(values)
    linear = dataset.rcs_to_linear(values)
    return dataset.linear_to_default_db(
        np.where(np.isfinite(linear) & (linear > 0), linear, np.nan),
        frequency_value=frequency, eps=0.0,
    )


def _plot_selection_indices(
    reference: RcsGrid,
    dataset: RcsGrid,
    azimuths,
    elevations,
    frequencies,
    polarization,
    *,
    native_axes=(),
):
    """Match fixed cuts exactly and preserve native samples along overlay sweeps."""

    from GRIM_Backend.plotting.modes import common as plot_common

    try:
        indices = []
        for axis, requested, values in (
            ("azimuth", azimuths, dataset.azimuths),
            ("elevation", elevations, dataset.elevations),
            ("frequency", frequencies, dataset.frequencies),
        ):
            if axis in native_axes:
                matched = plot_common.native_axis_selection(reference, dataset, axis, requested)
                if matched is None:
                    return None
                indices.append(matched.tolist())
            else:
                converted, tolerance = plot_common.selection_for_dataset(reference, dataset, axis, requested)
                indices.append(_indices(values, converted, tolerance=tolerance))
        return (*indices, _indices(dataset.polarizations, [polarization], text=True)[0])
    except (TypeError, ValueError):
        return None


def _plot_response_label(
    reference: RcsGrid,
    *,
    phase: bool,
    scale: str,
    p50: bool = False,
    mixed_db: bool = False,
) -> str:
    """Match the GUI's response-quantity labels in generated plots."""

    if phase:
        return "Phase P50 (deg)" if p50 else "Phase (deg)"
    if mixed_db:
        return "Mixed dB"
    quantity_name, linear_unit = {
        "sigma_3d": ("RCS", "m²"),
        "sigma_2d": ("Scattering Width", "m"),
        "power_ratio": ("Power Ratio", "dimensionless"),
        "ratio": ("Power Ratio", "dimensionless"),
    }.get(str(reference.linear_quantity()).strip().lower(), ("Value", "linear"))
    suffix = " P50" if p50 else ""
    if str(scale).strip().lower() == "linear":
        return f"{quantity_name}{suffix} ({linear_unit})"
    return f"{quantity_name}{suffix} ({reference.default_log_unit()})"


def plot_datasets(
    datasets: Sequence[tuple[str, RcsGrid]],
    *,
    mode: str,
    azimuths: Iterable[float],
    elevations: Iterable[float],
    frequencies: Iterable[float],
    polarization: str | Sequence[str],
    phase: bool = False,
    scale: str = "dbsm",
    colormap: str = "turbo",
    show_grid: bool = True,
    show_legend: bool = True,
    polar_zero: str = "N",
    reference_index: int = 0,
    waterfall_style: str = "Surface",
    isar_options: dict[str, object] | None = None,
    show_colorbar: bool = True,
    shared_colorbar: bool = True,
    square_aspect: bool = True,
    color_limits: tuple[float, float] | None = None,
    color_step: float = 0.0,
    delta_options: dict[str, object] | None = None,
    sweep_options: dict[str, object] | None = None,
    cdf_options: dict[str, object] | None = None,
    heatmap_axes: tuple[str, str] = ("azimuth", "frequency"),
    heatmap_aggregation: str = "panels",
    heatmap_cuts: bool = False,
    heatmap_options: dict[str, object] | None = None,
):
    """Create a Matplotlib Figure from resolved physical selector values.

    The implementation uses :class:`~matplotlib.backends.backend_agg.FigureCanvasAgg`
    directly and never imports pyplot or Qt.  Returned figures support ordinary
    ``figure.savefig(...)`` calls in generated scripts.

    Ordinary logarithmic overlays retain each dataset's native log convention;
    mixed quantities use a "Mixed dB" ordinate and identify units in the legend.
    Ordinary overlays accept one polarization or a sequence and preserve each
    dataset's native sweep samples within the selected reference intervals.
    ``sweep_options`` selects individual/mean/median/percentile/max azimuth
    reduction. ``cdf_options`` selects groups, weighting and an inclusive
    threshold; the returned figure exposes ``_grim_cdf_lookup_table``.
    ``heatmap_options`` selects axes, panels/mean_power and linked cuts.
    ``horizontal_cuts`` and ``vertical_cuts`` independently override the legacy
    ``linked_cuts`` flag. ``cut_position=[x, y]`` uses displayed physical units;
    each panel snaps to its nearest samples, without interpolation. Omission
    starts at the center sample of the first panel. Horizontal cuts appear
    below their image and share its X axis; vertical cuts appear to the right
    with response on X and the image's physical Y coordinate on Y.
    ``waterfall_style`` is retained for compatibility with existing scripts;
    Waterfall uses the physical two-axis image for every style value.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    selected = [(str(name), grid) for name, grid in datasets]
    if not selected:
        raise ValueError("plot_datasets needs at least one dataset")
    scale = str(scale).strip().lower()
    mode_key = canonical_plot_mode(mode)
    if mode_key not in HEADLESS_PLOT_MODES:
        raise ValueError(
            f"Unsupported headless plot mode {mode!r}; supported modes are "
            + ", ".join(sorted(HEADLESS_PLOT_MODES))
        )
    if mode_key == "isar_image" and len(selected) != 1:
        raise ValueError("ISAR headless plotting requires exactly one dataset")
    polarizations = tuple(dict.fromkeys(
        [polarization] if isinstance(polarization, str) else polarization
    ))
    if not polarizations or not all(isinstance(value, str) and value for value in polarizations):
        raise ValueError("polarization must contain one or more polarization names")
    if mode_key in {"delta_map", "isar_image"} and len(polarizations) != 1:
        raise ValueError(f"{mode_key} requires exactly one polarization")
    polarization = polarizations[0]
    try:
        reference_position = int(reference_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_index must identify one selected dataset") from exc
    if reference_position < 0 or reference_position >= len(selected):
        raise ValueError("reference_index must identify one selected dataset")
    reference = selected[reference_position][1]

    if mode_key == "delta_map":
        from GRIM_Backend.plotting.modes import delta_map_mode

        if phase or str(scale).strip().lower() == "linear":
            raise ValueError("Delta Map compares logarithmic levels in dB; phase and linear scales are unsupported")
        options = dict(delta_options or {})
        unknown = set(options) - {"x_axis", "y_axis", "limit", "show_values"}
        if unknown:
            raise ValueError(f"Unknown Delta Map options: {sorted(unknown)}")
        result = delta_map_mode.prepare(
            selected, reference=reference,
            selections={"azimuth": list(azimuths), "elevation": list(elevations), "frequency": list(frequencies)},
            polarization=polarization, x_axis=options.get("x_axis", "azimuth"),
            y_axis=options.get("y_axis", "frequency"),
        )
        figure = Figure(figsize=(10.0, 6.0), dpi=100, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        delta_map_mode.draw(figure, axes, result, limit=options.get("limit"),
                            show_values=bool(options.get("show_values", False)),
                            show_colorbar=bool(show_colorbar), colormap=colormap)
        figure.text(0.01, 0.01, f"A: {result.names[0]}   |   B: {result.names[1]}   |   A - B (dB)", fontsize=8, wrap=True)
        figure.set_layout_engine("tight", rect=(0, 0.07, 1, 1))
        return figure

    from GRIM_Backend.plotting.modes import common as plot_common

    plot_common.validate_plot_datasets(
        selected,
        phase=bool(phase),
        linear=str(scale).strip().lower() == "linear",
        allow_mixed_db=mode_key in {
            "azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep", "cdf", "waterfall"
        },
    )
    az_values = [float(value) for value in azimuths]
    el_values = [float(value) for value in elevations]
    freq_values = [float(value) for value in frequencies]
    if not az_values or not el_values or not freq_values:
        raise ValueError("azimuths, elevations, and frequencies cannot be empty")
    if mode_key == "waterfall":
        from GRIM_Backend.plotting.image_analysis import AXES, AXIS_ATTRIBUTES
        from GRIM_Backend.plotting.image_grid import line_with_gaps, native_sample_width
        from GRIM_Backend.plotting.image_figure import WaterfallFigure
        from matplotlib.colors import Normalize
        options = dict(heatmap_options or {})
        heatmap_axes = options.get("axes", heatmap_axes)
        heatmap_aggregation = options.get("aggregation", heatmap_aggregation)
        legacy_cuts = bool(options.get("linked_cuts", heatmap_cuts))
        horizontal_cuts = bool(options.get("horizontal_cuts", legacy_cuts))
        vertical_cuts = bool(options.get("vertical_cuts", legacy_cuts))
        cut_position = options.get("cut_position")
        if cut_position is not None:
            try:
                cut_position = np.asarray(cut_position, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Waterfall cut_position must contain two finite displayed coordinates [x, y]") from exc
            if cut_position.shape != (2,) or not np.all(np.isfinite(cut_position)):
                raise ValueError("Waterfall cut_position must contain two finite displayed coordinates [x, y]")
        x_axis, y_axis = tuple(heatmap_axes)
        if x_axis not in AXES or y_axis not in AXES or x_axis == y_axis:
            raise ValueError("Waterfall needs two different physical axes")
        if heatmap_aggregation not in {"panels", "mean_power"}:
            raise ValueError("Waterfall aggregation must be panels or mean_power")
        remaining = next(a for a in AXES if a not in (x_axis, y_axis))
        panels = []
        request = PlotRequest("waterfall", reference, (), phase=phase, scale=scale,
                              options={"axes": (x_axis, y_axis), "aggregation": heatmap_aggregation})
        for name, dataset in selected:
            for pol in polarizations:
                indices = _plot_selection_indices(reference, dataset, az_values, el_values, freq_values, pol)
                if indices is None:
                    continue
                indices = [*indices[:3], [indices[3]]]
                cuts = [None] if heatmap_aggregation == "mean_power" else indices[AXES.index(remaining)]
                x = plot_common.values_for_display(reference, dataset, x_axis, np.asarray(getattr(dataset, AXIS_ATTRIBUTES[x_axis]))[indices[AXES.index(x_axis)]])
                y = plot_common.values_for_display(reference, dataset, y_axis, np.asarray(getattr(dataset, AXIS_ATTRIBUTES[y_axis]))[indices[AXES.index(y_axis)]])
                xo, yo = np.argsort(x), np.argsort(y)
                indices[AXES.index(x_axis)] = np.asarray(indices[AXES.index(x_axis)])[xo]
                indices[AXES.index(y_axis)] = np.asarray(indices[AXES.index(y_axis)])[yo]
                plan = PlotPlan(name, dataset, tuple(indices))
                widths = (native_sample_width(reference, dataset, x_axis), native_sample_width(reference, dataset, y_axis))
                for cut in cuts:
                    prepared = prepare_waterfall_image(request, plan, x[xo], y[yo], remaining_index=cut,
                        x_width=widths[0], y_width=widths[1])
                    if prepared is None:
                        continue
                    snapshot = prepared["snapshot"]
                    plane = snapshot[2].values if isinstance(snapshot, tuple) else prepared["native_display"]
                    suffix = (f"{'Circular phase mean' if phase else 'Mean linear display quantity'} over {len(indices[AXES.index(remaining)])} {remaining} samples" if cut is None else
                              f"{remaining.capitalize()} {plot_common.values_for_display(reference, dataset, remaining, [getattr(dataset, AXIS_ATTRIBUTES[remaining])[cut]])[0]:g} {plot_common.axis_unit(reference, remaining)}")
                    response = _plot_response_label(dataset, phase=phase, scale=scale)
                    panels.append((f"{name} | Pol {pol} | {suffix} | {response}", x[xo], y[yo], plane, response, prepared))
        if not panels:
            raise ValueError("No compatible Waterfall coordinates")
        figure = WaterfallFigure(figsize=(10., max(4., 4.*len(panels))), dpi=100, facecolor="white")
        FigureCanvasAgg(figure)
        figure._grim_waterfall_panels = [panel[5] for panel in panels]
        figure.canvas.mpl_connect("close_event", figure._release_waterfall)
        cut_count = int(horizontal_cuts) + int(vertical_cuts)
        shared_bar = bool(show_colorbar and shared_colorbar)
        spec = figure.add_gridspec(
            len(panels), 2 if shared_bar else 1,
            width_ratios=[1, .035] if shared_bar else None,
        )
        if cut_position is None:
            first_x, first_y = panels[0][1:3]
            cut_position = (first_x[len(first_x) // 2], first_y[len(first_y) // 2])
        norms = {}
        for _title, _x, _y, data, response, prepared in panels:
            display_values = prepared["display"]
            finite = display_values[np.isfinite(display_values)]
            if finite.size:
                old = norms.get(response, (np.inf, -np.inf))
                norms[response] = (min(old[0], float(np.min(finite))), max(old[1], float(np.max(finite))))
        callbacks, image_axes, meshes, cut_views = [], [], [], []
        global_limits = (min(v[0] for v in norms.values()), max(v[1] for v in norms.values())) if norms else None
        for i, (title, x, y, data, response, prepared) in enumerate(panels):
            individual_bar = bool(show_colorbar and not shared_colorbar)
            # Marginal cuts occupy the same grid column/row as their image.
            # Dedicated colorbar cells preserve that alignment after layout.
            panel_spec = spec[i, 0].subgridspec(
                2 if horizontal_cuts else 1,
                1 + int(vertical_cuts) + int(individual_bar),
                height_ratios=[3, 1] if horizontal_cuts else None,
                width_ratios=[4] + ([1.4] if vertical_cuts else [])
                             + ([.16] if individual_bar else []),
            )
            ax = figure.add_subplot(panel_spec[0, 0])
            limits = color_limits if color_limits is not None else global_limits if shared_colorbar else None
            norm = Normalize(*limits) if limits is not None else None
            xe, ye, display = prepared["x_edges"], prepared["y_edges"], prepared["drawn"]
            mesh = ax.pcolormesh(xe, ye, display,
                                 shading="flat", cmap=colormap, norm=norm)
            ax.set(title=title, xlabel=plot_common.axis_label(reference, x_axis), ylabel=plot_common.axis_label(reference, y_axis))
            ax.grid(bool(show_grid))
            image_axes.append(ax); meshes.append(mesh)
            if cut_count:
                ix = int(np.argmin(abs(x - cut_position[0])))
                iy = int(np.argmin(abs(y - cut_position[1])))
                horizontal = vertical = row = column = hline = vline = None
                if horizontal_cuts:
                    horizontal = figure.add_subplot(panel_spec[1, 0], sharex=ax)
                    row, = horizontal.plot(*line_with_gaps(x, data[:, iy]))
                    hline = ax.axhline(y[iy], color="white", linewidth=.8)
                    horizontal.set(xlabel=plot_common.axis_label(reference, x_axis), ylabel=response,
                                   title=f"{y_axis} = {y[iy]:g} {plot_common.axis_unit(reference, y_axis)}")
                    horizontal.grid(bool(show_grid))
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)
                if vertical_cuts:
                    vertical = figure.add_subplot(panel_spec[0, 1], sharey=ax)
                    coordinates, values = line_with_gaps(y, data[ix, :])
                    column, = vertical.plot(values, coordinates)
                    vline = ax.axvline(x[ix], color="white", linewidth=.8)
                    vertical.set(xlabel=response,
                                 title=f"{x_axis} = {x[ix]:g} {plot_common.axis_unit(reference, x_axis)}")
                    vertical.grid(bool(show_grid))
                    vertical.tick_params(axis="y", labelleft=False)
                cut_views.append((x, y, data, horizontal, vertical, row, column, hline, vline))
            ax.set_xlim(xe[0], xe[-1])
            ax.set_ylim(ye[0], ye[-1])
            if individual_bar:
                color_axis = figure.add_subplot(panel_spec[0, -1])
                figure.colorbar(mesh, cax=color_axis, label=response)
        if cut_views:
            def update(event):
                if (event.inaxes not in image_axes or event.button != 1
                        or event.xdata is None or event.ydata is None
                        or not np.isfinite(event.xdata) or not np.isfinite(event.ydata)):
                    return
                toolbar = getattr(figure.canvas, "toolbar", None)
                if toolbar is not None and getattr(toolbar, "mode", None):
                    return
                for x, y, data, horizontal, vertical, row, column, hline, vline in cut_views:
                    ix = int(np.argmin(abs(x - event.xdata)))
                    iy = int(np.argmin(abs(y - event.ydata)))
                    if horizontal is not None:
                        row.set_data(*line_with_gaps(x, data[:, iy]))
                        hline.set_ydata([y[iy], y[iy]])
                        horizontal.set_title(f"{y_axis} = {y[iy]:g} {plot_common.axis_unit(reference, y_axis)}")
                        horizontal.relim(); horizontal.autoscale_view(scalex=False)
                    if vertical is not None:
                        coordinates, values = line_with_gaps(y, data[ix, :])
                        column.set_data(values, coordinates)
                        vline.set_xdata([x[ix], x[ix]])
                        vertical.set_title(f"{x_axis} = {x[ix]:g} {plot_common.axis_unit(reference, x_axis)}")
                        vertical.relim(); vertical.autoscale_view(scaley=False)
                figure.canvas.draw_idle()
            callbacks.append(figure.canvas.mpl_connect("button_press_event", update))
        if show_colorbar and shared_colorbar:
            response = panels[0][4] if len({p[4] for p in panels}) == 1 else "Mixed dB"
            color_axis = figure.add_subplot(spec[:, 1])
            figure.colorbar(meshes[-1], cax=color_axis, label=response)
        figure._grim_heatmap_cut_callbacks = callbacks
        if any(panel[5]["snapshot"] is not None for panel in panels):
            figure.text(.01, .005, "Image shows block maxima; cuts use original samples.", fontsize=8)
        figure.set_layout_engine("constrained")
        return figure

    if mode_key == "cdf":
        from GRIM_Backend.plotting import distribution_analysis as distribution
        if phase:
            raise ValueError("CDF plots rank magnitude levels, not phase")
        config = distribution.options(cdf_options)
        plans = []
        for name, dataset in selected:
            for pol in polarizations:
                indices = _plot_selection_indices(reference, dataset, az_values, el_values, freq_values, pol)
                if indices is None:
                    continue
                plans.append(PlotPlan(name, dataset, (*indices[:3], [indices[3]])))
        request = PlotRequest("cdf", reference, tuple(plans), scale=scale, options=config,
                              selectors={"azimuth": az_values, "frequency": freq_values})
        curves = prepare_cdf(request).items
        if not curves:
            raise ValueError("No valid magnitude population for the selected CDF")
        figure = Figure(figsize=(10., 6.), dpi=100, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        limits = distribution.bounds(curves, config["threshold"] if config["threshold_enabled"] else None)
        mixed = len({(c["dataset"].linear_quantity(), c["unit"]) for c in curves}) > 1
        for curve in curves:
            x, y = distribution.draw_values(curve, limits, exceedance=config["exceedance"])
            label = curve["label"]
            if mixed:
                head, separator, tail = label.partition(" | ")
                label = f"{head} [{curve['unit']}]{separator}{tail}"
            line, = axes.plot(x, y, label=label, drawstyle="steps-pre" if config["exceedance"] else "steps-post")
            line._grim_cdf_population = curve["population"]["counts"]
            line._grim_cdf_threshold_lookup = curve["lookup"]
            line._grim_trace_key = curve["key"]
        if config["threshold_enabled"]:
            axes.axvline(config["threshold"], color="0.4", linestyle=":", label="_CDF threshold")
            axes.text(.01, .98, "Threshold lookup (inclusive):\n" + "\n".join(
                f"{c['name']} | {c['pol']} | {c['group']}: <= {c['lookup'][0]:.4g}%, >= {c['lookup'][1]:.4g}% at {config['threshold']:g} {c['unit']}"
                for c in curves), transform=axes.transAxes, va="top", fontsize=8)
        notes = []
        if any(np.any(np.isneginf(c["levels"])) for c in curves):
            notes.append("Zero power retained at -infinity dB.")
        if config["weighting"] != "samples":
            notes.append("Trapezoidal measure over selected group coordinate hull, including gaps; not dwell time or solid angle.")
        if notes:
            figure.text(.01, .01, " ".join(notes), fontsize=8)
        axes.set_xlabel(_plot_response_label(curves[0]["dataset"], phase=False, scale=scale, mixed_db=mixed))
        axes.set_ylabel(distribution.y_label(config))
        axes.set_ylim(0., 100.)
        axes.grid(bool(show_grid))
        if show_legend:
            axes.legend()
        figure._grim_cdf_lookup_table = distribution.table_text(curves, config)
        figure.tight_layout(rect=(0, .05 if notes else 0, 1, 1))
        return figure

    compatible_count = 0
    style_axes = None
    plotted_series = []
    overlay_series = [
        (name, dataset, pol)
        for name, dataset in selected
        for pol in polarizations
    ]

    figure = Figure(figsize=(10.0, 6.0), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    if mode_key == "azimuth_polar":
        axes = figure.add_subplot(111, projection="polar")
        axes.set_theta_zero_location(str(polar_zero))
        axes.set_theta_direction(-1)
    else:
        axes = figure.add_subplot(111)

    if mode_key in {"azimuth_rect", "azimuth_polar"}:
        for name, dataset, polarization in overlay_series:
            indices = _plot_selection_indices(
                reference,
                dataset,
                az_values,
                el_values,
                freq_values,
                polarization,
                native_axes=("azimuth",),
            )
            if indices is None:
                continue
            az_idx, el_idx, fr_idx, pol_idx = indices
            compatible_count += 1
            native_x = np.asarray(dataset.azimuths)[az_idx]
            x = plot_common.values_for_display(
                reference, dataset, "azimuth", native_x
            )
            order = np.argsort(x)
            x = np.asarray(x)[order]
            breaks = plot_common.selection_breaks(
                reference, dataset, "azimuth", np.asarray(az_idx)[order], az_values)
            if mode_key == "azimuth_polar":
                x = plot_common.convert_axis_values(
                    x,
                    "azimuth",
                    plot_common.axis_unit(reference, "azimuth"),
                    "rad",
                )
            for fi in fr_idx:
                for ei in el_idx:
                    selection = np.ix_(az_idx, [ei], [fi], [pol_idx])
                    raw = (
                        dataset.rcs_slice(selection)[:, 0, 0, 0]
                        if phase
                        else dataset.power[np.ix_(az_idx, [ei], [fi], [pol_idx])][:, 0, 0, 0]
                    )
                    y = _display_values(
                        dataset,
                        raw,
                        frequency=float(dataset.frequencies[fi]),
                        phase=phase,
                        scale=scale,
                    )[order]
                    draw_x, draw_y = plot_common.insert_plot_breaks(breaks, x, y)
                    lines = axes.plot(
                        draw_x,
                        draw_y,
                        **plot_common.singleton_line_markers(draw_x, draw_y),
                        linestyle="-",
                        label=(
                            f"{name} | Pol {polarization}, Freq "
                            f"{float(plot_common.values_for_display(reference, dataset, 'frequency', [dataset.frequencies[fi]])[0]):.12g} "
                            f"{plot_common.axis_unit(reference, 'frequency')}, "
                            f"{plot_common.angular_axis_name(reference, 'elevation')} "
                            f"{float(plot_common.values_for_display(reference, dataset, 'elevation', [dataset.elevations[ei]])[0]):.12g} "
                            f"{plot_common.axis_unit(reference, 'elevation')}"
                        ),
                    )
                    plotted_series.extend((line, dataset) for line in lines)
        axes.set_xlabel(plot_common.axis_label(reference, "azimuth"))
        axes.set_ylabel(
            _plot_response_label(reference, phase=phase, scale=scale)
        )

    elif mode_key in {"frequency", "elevation_sweep"}:
        from GRIM_Backend.plotting import sweep_analysis
        config = sweep_analysis.options(sweep_options)
        sweep_axis = "frequency" if mode_key == "frequency" else "elevation"
        plans = []
        for name, dataset, polarization in overlay_series:
            indices = _plot_selection_indices(
                reference, dataset, az_values, el_values, freq_values, polarization,
                native_axes=("azimuth", sweep_axis),
            )
            if indices is None:
                continue
            compatible_count += 1
            plans.append(PlotPlan(name, dataset, (*indices[:3], [indices[3]])))
        request = PlotRequest(mode_key, reference, tuple(plans), phase=phase, scale=scale,
                              options=config)
        for plan, (x, y, label, key) in prepare_sweep(request).items:
            dataset, indices = plan.dataset, plan.selection
            order = np.argsort(x)
            sweep_indices = indices[2] if sweep_axis == "frequency" else indices[1]
            requested = freq_values if sweep_axis == "frequency" else el_values
            breaks = plot_common.selection_breaks(
                reference, dataset, sweep_axis, np.asarray(sweep_indices)[order], requested)
            draw_x, draw_y = plot_common.insert_plot_breaks(
                breaks, np.asarray(x)[order], np.asarray(y)[order])
            lines = axes.plot(draw_x, draw_y, linestyle="-", label=label,
                              **plot_common.singleton_line_markers(draw_x, draw_y))
            for line in lines:
                line._grim_trace_key = key
            plotted_series.extend((line, dataset) for line in lines)
        axes.set_xlabel(plot_common.axis_label(reference, sweep_axis))
        response = _plot_response_label(reference, phase=phase, scale=scale)
        tag = sweep_analysis.axis_tag(config, phase=phase) if mode_key == "frequency" or len(az_values) > 1 or config["mode"] != "median" else ""
        axes.set_ylabel(response.replace(" (", tag + " (", 1))

    elif mode_key == "isar_image":
        from GRIM_Backend.plotting.modes.isar_mode import _length_unit, form_isar
        from GRIM_Backend.isar.geometry import image_extent, draw_image, image_axis_labels
        from GRIM_Backend.isar.quality import reconstruction_advisories

        dataset = selected[0][1]
        options = dict(isar_options or {})
        options.setdefault("length_unit", "in")


        options["decimate_display"] = True
        options["retain_complex"] = False
        az_idx = _indices(dataset.azimuths, az_values)
        fr_idx = _indices(dataset.frequencies, freq_values)
        el_idx = _indices(dataset.elevations, [el_values[0]])[0]
        pol_idx = _indices(dataset.polarizations, [polarization], text=True)[0]
        bands, _elapsed = form_isar(
            dataset,
            azimuth_indices=az_idx,
            frequency_indices=fr_idx,
            elevation_index=el_idx,
            polarization_index=pol_idx,
            **options,
        )
        compatible_count = 1
        figure.clear()
        # Match the GUI's centered fixed-aspect image/colorbar group.
        figure.set_layout_engine("compressed" if square_aspect else "constrained")
        axes_values = figure.subplots(1, len(bands), squeeze=False)[0]
        style_axes = list(axes_values)
        displays = []
        for band in bands:
            magnitude = np.asarray(band["magnitude"], dtype=np.float32)
            intensity = np.empty_like(magnitude, dtype=np.float32)
            np.multiply(magnitude, magnitude, out=intensity)
            if str(scale).strip().lower() != "linear":


                np.maximum(intensity, np.float32(1.0e-12), out=intensity)
                np.log10(intensity, out=intensity)
                intensity *= np.float32(10.0)
            displays.append(intensity)

        clamp = None
        if color_limits is not None:
            values = np.asarray(color_limits, dtype=float).reshape(-1)
            if (
                values.size != 2
                or not np.all(np.isfinite(values))
                or not values[0] < values[1]
            ):
                raise ValueError("color_limits must be two finite increasing values")
            clamp = (float(values[0]), float(values[1]))
        shared_limits = (
            plot_common.finite_data_limits(displays)
            if bool(shared_colorbar) and clamp is None
            else None
        )
        plot_vmin = clamp[0] if clamp is not None else (
            shared_limits[0] if shared_limits is not None else None
        )
        plot_vmax = clamp[1] if clamp is not None else (
            shared_limits[1] if shared_limits is not None else None
        )

        meshes = []
        for axis, band, display in zip(axes_values, bands, displays):
            mesh = draw_image(
                axis, band, display,
                cmap=colormap,
                vmin=plot_vmin,
                vmax=plot_vmax,
            )
            if bool(square_aspect):
                # Match the GUI: redraws must preserve the requested limits.
                axis.set_aspect("equal", adjustable="box")
            meshes.append(mesh)

        unit_name, _unit_scale = _length_unit(options["length_unit"])
        elevation_native = float(np.asarray(dataset.elevations)[el_idx])
        elevation_deg = float(
            plot_common.convert_axis_values(
                [elevation_native],
                "elevation",
                plot_common.axis_unit(dataset, "elevation"),
                "deg",
            )[0]
        )
        elevation_name = plot_common.angular_axis_name(dataset, "elevation")
        reconstruction = str(options.get("reconstruction", "fast")).strip().lower()
        resolved = {band.get('resolved_reconstruction', reconstruction) for band in bands}
        if len(resolved) > 1:
            reconstruction_label = "PFA method selected per aperture"
        elif 'sparse' in resolved or reconstruction in {"sparse", "l1", "sparse-l1"}:
            reconstruction_label = "Sparse L1 (Experimental)"
        elif 'accurate' in resolved or reconstruction in {"accurate", "cartesian", "pfa-accurate"}:
            reconstruction_label = "Cartesian PFA"
        else:
            reconstruction_label = "Fast PFA"
        composite_sublooks = max(
            (int(band.get("composite", 0)) for band in bands), default=0
        )
        if composite_sublooks:
            if all(band.get("composite") for band in bands):
                reconstruction_label += (
                    f" | Wide-Aperture Composite ({composite_sublooks} looks)"
                )
            else:
                reconstruction_label += " | Coherent + Wide-Aperture Composite"
        title = (
            f"ISAR Image | {elevation_name} {elevation_deg:g} deg | "
            f"Pol {dataset.polarizations[pol_idx]} | {reconstruction_label}"
        )
        if len(bands) > 1:
            figure.suptitle(title)
            for axis, band in zip(axes_values, bands):
                axis.set_title(
                    f"{float(band['az_values'][0]):g}°–"
                    f"{float(band['az_values'][-1]):g}°"
                )
        else:
            axes_values[0].set_title(title)

        for axis, band in zip(axes_values, bands):
            x_label, y_label = image_axis_labels(
                band, unit_name, elevation_deg, explicit_frame=len(bands) > 1
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)

        colorbars = []
        if bool(show_colorbar) and meshes:
            if bool(shared_colorbar):
                colorbars.append(figure.colorbar(meshes[-1], ax=list(axes_values)))
            else:
                colorbars.extend(
                    figure.colorbar(mesh, ax=axis)
                    for axis, mesh in zip(axes_values, meshes)
                )
            label = (
                "Image Intensity (linear)"
                if str(scale).strip().lower() == "linear"
                else "Image Intensity (dB)"
            )
            try:
                tick_step = float(color_step)
            except (TypeError, ValueError) as exc:
                raise ValueError("color_step must be finite and nonnegative") from exc
            if not np.isfinite(tick_step) or tick_step < 0.0:
                raise ValueError("color_step must be finite and nonnegative")
            for colorbar in colorbars:
                colorbar.set_label(label)
                if tick_step > 0.0:
                    low, high = colorbar.mappable.get_clim()
                    first = math.ceil(low / tick_step) * tick_step
                    tick_count = int(math.floor((high - first) / tick_step)) + 1
                    if 0 < tick_count <= 1000:
                        colorbar.set_ticks(first + tick_step * np.arange(tick_count))

        for advisory in reconstruction_advisories(bands):
            warnings.warn(advisory, RuntimeWarning, stacklevel=2)
        sparse_results = [
            band for band in bands if band.get("sparse_iterations") is not None
        ]
        if sparse_results and not all(
            bool(band.get("sparse_converged", False)) for band in sparse_results
        ):
            worst_gap = max(
                float(band.get("sparse_relative_duality_gap", float("inf")))
                for band in sparse_results
            )
            warnings.warn(
                "ISAR sparse reconstruction did not certify convergence for "
                f"every band (worst relative duality gap {worst_gap:.3g}); "
                "treat the image as diagnostic, not quantitatively final.",
                RuntimeWarning,
                stacklevel=2,
            )
        axes = axes_values[0]
    else:
        raise ValueError(f"Unsupported plot mode: {mode!r}")

    if compatible_count == 0:
        figure.clear()
        raise ValueError(
            "None of the selected datasets has compatible samples for the requested cuts and polarizations"
        )

    if plotted_series and not phase and str(scale).strip().lower() != "linear":
        # Match the GUI's visible-artist policy: a selected dataset whose
        # coordinates cannot be plotted must not change the axis or legend.
        mixed_db = len({
            (
                str(dataset.linear_quantity()).strip().lower(),
                dataset.default_log_unit().strip().lower(),
            )
            for _, dataset in plotted_series
        }) > 1
        response = _plot_response_label(plotted_series[0][1], phase=False, scale=scale, mixed_db=mixed_db)
        if mode_key in {"frequency", "elevation_sweep"}:
            from GRIM_Backend.plotting import sweep_analysis
            config = sweep_analysis.options(sweep_options)
            tag = sweep_analysis.axis_tag(config) if mode_key == "frequency" or len(az_values) > 1 or config["mode"] != "median" else ""
            response = response if mixed_db else response.replace(" (", tag + " (", 1)
        axes.set_ylabel(response)
        if mixed_db:
            for line, dataset in plotted_series:
                head, separator, tail = line.get_label().partition(" | ")
                line.set_label(
                    f"{head} [{dataset.default_log_unit()}]{separator}{tail}"
                )

    for axis in (style_axes if style_axes is not None else figure.axes):
        axis.grid(bool(show_grid))
        handles, labels = axis.get_legend_handles_labels()
        if show_legend and handles:
            axis.legend()
    if mode_key != "isar_image":
        figure.tight_layout()
    return figure
