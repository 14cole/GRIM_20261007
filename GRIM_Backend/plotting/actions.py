from __future__ import annotations

import threading
from uuid import uuid4

import numpy as np

from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QListWidget,
    QToolButton,
)

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.dataset_style import (
    PBP_BAND_KEY,
    DatasetPlotStyleMixin,
    is_pbp_band_key,
    pbp_band_key,
)
from GRIM_Backend.plotting.markers import PlotMarkersMixin
from GRIM_Backend.plotting.modes import (
    az_vs_range_mode,
    azimuth_polar_mode,
    azimuth_rect_mode,
    cdf_mode,
    compare_mode,
    delta_map_mode,
    elevation_sweep_mode,
    frequency_mode,
    isar_mode,
    range_freq_mode,
    sector_stats_mode,
    waterfall_mode,
)
from GRIM_Backend.plotting.modes import common as plot_common
from GRIM_Backend.plotting.slider import PlotSliderMixin


class _IsarComputeSignals(QObject):
    """Bridges the ISAR worker thread back to the GUI thread. Emitting from
    the worker queues the slot call onto the GUI event loop (cross-thread
    signals auto-queue), so all Qt/matplotlib work stays on the GUI thread."""

    done = Signal(object, object)  # (params, result)
    progress = Signal(object, str)


_AXIS_AVAILABILITY_WORK_BYTES = 8 * 1024**2

NATIVE_DB_LINE_MODES = (
    "azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep",
    "cdf", "sector_stats",
)


class _PbpBands:
    """One streaming envelope per dataset and polarization for this render."""

    def __init__(self, owner, datasets):
        self._owner = owner
        self._names = {owner._dataset_plot_key(dataset): name for name, dataset in datasets}
        self._envelopes = {}
        self._db_quantities = {}

    def update(self, dataset, values, *, polarization=None) -> None:
        identity = (self._owner._dataset_plot_key(dataset), str(polarization or ""))
        envelope = self._envelopes.get(identity)
        if envelope is None:
            envelope = self._envelopes[identity] = self._owner._new_pbp_envelope()
        envelope.update(values)
        if self._owner._native_db_overlay_enabled():
            self._db_quantities.setdefault(identity, set()).add(
                self._owner._native_db_quantity(dataset)
            )

    def accept_prepared(self, plan, envelope):
        """Adopt already reduced worker data without retaining every trace."""
        dataset = plan.dataset
        identity = (self._owner._dataset_plot_key(dataset), str(dataset.polarizations[plan.selection[3][0]]))
        self._envelopes[identity] = envelope
        if self._owner._native_db_overlay_enabled():
            self._db_quantities.setdefault(identity, set()).add(self._owner._native_db_quantity(dataset))

    def _band_key(self):
        """A held render is a new plot item, even for the same dataset/cuts."""
        if not self._owner._plot_item_artists(PBP_BAND_KEY):
            return PBP_BAND_KEY
        return pbp_band_key(uuid4().hex)

    def draw(self, x_values, description: str, *, polar: bool, to_plot_x=None, break_before=()) -> None:
        owner = self._owner
        held_keys = {
            artist._grim_dataset_key
            for artist in (*owner.plot_ax.lines, *owner.plot_ax.collections)
            if is_pbp_band_key(getattr(artist, "_grim_dataset_key", None))
        }
        multiple = len(held_keys) + len(self._envelopes) > 1
        heatmap = not multiple and owner.pbp_fill_mode in ("heatmap_rcs", "heatmap_density")
        if multiple and owner.pbp_fill_mode in ("heatmap_rcs", "heatmap_density"):
            owner._note_plot_render(
                "New PBP bands use translucent plot colors when several bands "
                "are displayed; heatmap fills apply to a single band."
            )
        # Preserve dataset selection order, then polarization order, just as
        # ordinary curves do. Each complete band consumes one line-cycle color.
        for identity, envelope in self._envelopes.items():
            if envelope.lower is None:
                envelope.close()
                continue
            lower, upper, density = envelope.result()
            percentiles = envelope.percentiles
            envelope.close()
            x_segmented, lower_segmented, upper_segmented, density_segmented = plot_common.insert_plot_breaks(
                break_before, x_values, lower, upper, density)
            if envelope.phase_degrees:
                lower_segmented, upper_segmented = plot_common.continuous_phase_envelope(
                    lower_segmented, upper_segmented)
            # Probes keep original sample positions; drawing adds separators.
            native_mask = np.isfinite(x_segmented)
            native_x = x_segmented[native_mask]
            native_edges = (to_plot_x(native_x) if to_plot_x is not None else native_x,
                            lower_segmented[native_mask], upper_segmented[native_mask])
            zero_notice = None
            if np.any(np.isneginf(lower)) or np.any(np.isneginf(upper)):
                owner._note_plot_render(
                    "PBP includes zero returns as −∞ dB. Unbounded fill is omitted; finite edges remain visible."
                )
                if not np.any(np.isfinite(lower) | np.isfinite(upper)):
                    zero_notice = owner.plot_ax.text(.5, .5, "PBP bounds: −∞ dB (zero power)",
                                                    ha="center", va="center", transform=owner.plot_ax.transAxes)
            x_display, lower, upper, density_display = owner._bounded_plot_envelope(
                x_segmented, lower_segmented, upper_segmented, density_segmented
            )
            if to_plot_x is not None:
                x_display = to_plot_x(x_display)
            prefix = (
                "PBP" if percentiles is None
                else f"PBP P{percentiles[0]:g}–P{percentiles[1]:g}"
            )
            name = self._names[identity[0]]
            band_description = description.removeprefix(f"Pol {identity[1]}, ")
            label = f"{prefix} {name} | Pol {identity[1]}, {band_description}"
            color = owner.plot_ax._get_lines.get_next_color()
            key = self._band_key()
            if zero_notice is not None:
                zero_notice._grim_dataset_key = key
            owner._plot_pbp_band(
                x_display, lower, upper, label, polar, density=density_display,
                key=key, color=color, heatmap=heatmap, native_edges=native_edges,
            )
            for artist in owner._plot_item_artists(key):
                artist._grim_pbp_identity = identity
                artist._grim_pbp_count = np.asarray(density)
                owner._tag_native_db_artist(artist, self._db_quantities.get(identity, set()))


def _selected_polarization_axis_availability(
    dataset: RcsGrid,
    polarization_indices,
    *,
    require_phase: bool,
    maximum_work_bytes: int = _AXIS_AVAILABILITY_WORK_BYTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-sample masks for frequency, elevation, and azimuth.

    The parameter sidebar asks this question whenever polarization selection
    changes.  Advanced-indexing the polarization axis first copied an entire
    selected 4-D power grid (and another phase grid in phase mode).  Scan
    basic-slice azimuth blocks instead: input arrays remain views and the only
    transient is one bounded Boolean block.
    """

    shape = tuple(int(value) for value in dataset.power.shape)
    if len(shape) != 4 or tuple(dataset.phase.shape) != shape:
        raise ValueError("dataset power and phase grids must have matching 4-D shapes")
    selected = sorted({int(value) for value in polarization_indices})
    if not selected:
        return (
            np.zeros(shape[2], dtype=bool),
            np.zeros(shape[1], dtype=bool),
            np.zeros(shape[0], dtype=bool),
        )
    if selected[0] < 0 or selected[-1] >= shape[3]:
        raise IndexError("polarization index is out of range")
    budget = int(maximum_work_bytes)
    if budget < 1:
        raise ValueError("maximum_work_bytes must be positive")

    frequency_available = np.zeros(shape[2], dtype=bool)
    elevation_available = np.zeros(shape[1], dtype=bool)
    azimuth_available = np.zeros(shape[0], dtype=bool)
    cells_per_azimuth = max(1, shape[1] * shape[2])
    simultaneous_masks = 2 if require_phase else 1
    azimuth_block = max(
        1,
        min(
            shape[0],
            budget // (cells_per_azimuth * simultaneous_masks),
        ),
    )

    for pol_index in selected:
        for start in range(0, shape[0], azimuth_block):
            stop = min(shape[0], start + azimuth_block)
            selection = (slice(start, stop), slice(None), slice(None), pol_index)
            valid = np.isfinite(dataset.power[selection])
            if require_phase:
                valid &= np.isfinite(dataset.phase[selection])
            azimuth_available[start:stop] |= valid.any(axis=(1, 2))
            elevation_available |= valid.any(axis=(0, 2))
            frequency_available |= valid.any(axis=(0, 1))

    return frequency_available, elevation_available, azimuth_available


class PlotOpsMixin(PlotMarkersMixin, PlotSliderMixin, DatasetPlotStyleMixin):
    def _cancel_plot_preparations(self):
        from GRIM_Backend.plotting.modes.cdf_mode import clear_state
        canceled = False
        for context in getattr(self, "_plot_contexts", {}).values():
            clear_state(context.plot_figure)
            panel = getattr(context, 'inline_analysis', None)
            if panel is not None:
                panel._analysis_runner.cancel()
                panel._cut_cache = {}
            canceled = context.session.cancel() or canceled
        if canceled:
            self.status.showMessage("Pending plot canceled because its inputs changed.")

    def _submit_plot_preparation(self, request, compute, publish, *, cells):
        """Prepare substantial array work separately from GUI publication.

        Small selections stay immediate; this keeps a click on an ordinary
        short cut inexpensive. Tests and non-window callers use the same pure
        service synchronously. Large jobs are coalesced per owning canvas.
        """
        contexts = getattr(self, "_plot_contexts", {})
        context = contexts.get(getattr(self, "_active_plot_tab", None))
        threshold = int(getattr(self, "_plot_async_threshold", 100_000))
        if context is None or int(cells) < threshold:
            try:
                publish(compute(None))
            except (MemoryError, ValueError) as exc:
                self.status.showMessage(f"Plot preparation failed: {exc}")
            return False
        from GRIM_Backend.execution.plot_jobs import PlotJobRunner, dispose_result
        session = context.session
        if session.runner is None:
            session.runner = PlotJobRunner(self)
            session.runner.publication_failed.connect(
                lambda error: self.status.showMessage(f"Plot publication failed: {error}"))
        generation = int(getattr(self, "_plot_render_generation", 0))
        tab_key = self._active_plot_tab
        old_recipe = getattr(self, "last_python_plot_spec", None)
        self._record_python_plot(request.mode, emit=False)
        recipe = self.last_python_plot_spec
        self.last_python_plot_spec = old_recipe
        self._ordinary_plot_pending = True
        self._ordinary_plot_emit = False
        self._ordinary_plot_fit = None
        self._ordinary_plot_slider_before = None
        self.status.showMessage(f"Preparing {request.mode.replace('_', ' ')} plot…")

        def completed(result, error):
            if (session.disposed or getattr(self, "_active_plot_tab", None) != tab_key
                    or context.plot_figure is not self.plot_figure
                    or generation != int(getattr(self, "_plot_render_generation", 0))):
                dispose_result(result)
                return
            self._ordinary_plot_pending = False
            if error is not None:
                self.status.showMessage(f"Plot preparation failed: {error}")
                return
            if result is None:
                return
            try:
                publish(result)
            except Exception as exc:
                dispose_result(result)
                self.status.showMessage(f"Plot display failed: {exc}")
                return
            if "updated" in str(self.status.currentMessage()).lower():
                self.last_python_plot_spec = recipe
                self._update_current_python_plot_style()
                slider_before = self._ordinary_plot_slider_before
                if slider_before is not None:
                    self._tag_plot_slider_artists(slider_before)
                self._restore_plot_markers()
                fit = getattr(self, "_ordinary_plot_fit", None)
                if fit:
                    getattr(self, fit)()
                self._maybe_autoscale()
                if self._ordinary_plot_emit:
                    self._emit_last_successful_python_plot()
        session.runner.submit(compute, completed)
        return True

    def _on_param_selection_changed(self) -> None:
        self._invalidate_isar_result()
        self._maybe_autoplot()

    def _on_polarization_selection_changed(self) -> None:
        self._invalidate_isar_result()
        if self.active_dataset is None:
            return
        selected_pol = sorted(self._selected_indices(self.list_pol))
        if not selected_pol:
            self._sync_axis_list(self.list_freq, self.active_dataset.frequencies, None)
            self._sync_axis_list(self.list_elev, self.active_dataset.elevations, None)
            self._sync_axis_list(self.list_az, self.active_dataset.azimuths, None)
            return

        freq_available, elev_available, az_available = (
            _selected_polarization_axis_availability(
                self.active_dataset,
                selected_pol,
                require_phase=self._button_checked(self.btn_phase),
            )
        )

        self._sync_axis_list(
            self.list_freq, self.active_dataset.frequencies, freq_available
        )
        self._sync_axis_list(
            self.list_elev, self.active_dataset.elevations, elev_available
        )
        self._sync_axis_list(
            self.list_az, self.active_dataset.azimuths, az_available
        )
        self._maybe_autoplot()

    def _sync_axis_list(self, widget, values, avail_mask) -> None:
        """Refill an axis list only if the set of displayed indices changed.

        Skipping unchanged rebuilds preserves selection without reselect calls
        and avoids the QListWidget churn that dominates UI lag for large axes
        (e.g. 1601 frequency samples).
        """
        if avail_mask is None:
            new_indices = set(range(len(values)))
            new_index_list = list(range(len(values)))
        else:
            new_index_list = [int(i) for i in np.where(avail_mask)[0]]
            new_indices = set(new_index_list)
        if self._displayed_indices(widget) == new_indices:
            return
        prev_selection = self._selected_indices(widget)
        self._fill_list(widget, values, new_index_list)
        self._reselect_indices(widget, prev_selection)

    def _maybe_autoplot(self) -> None:
        if not self._button_checked(self.btn_auto_plot):
            return
        # Hold is an explicit overlay workflow. Selection bursts under Auto
        # Plot would otherwise append transient/intermediate series as the user
        # ctrl-clicks a new cut, even though they never requested those traces.
        if self._button_checked(getattr(self, "btn_hold", None)):
            return
        if self.last_plot_mode is None:
            return
        # Debounce burst selection events (shift-click, ctrl-A) so a tight
        # sequence collapses into a single render. 50 ms feels responsive
        # but groups bursts; matters most for ISAR with many freq samples.
        timer = getattr(self, "_autoplot_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._do_autoplot)
            self._autoplot_timer = timer
        timer.start(50)

    def _do_autoplot(self) -> None:
        if self.last_plot_mode is None:
            return
        self._render_plot_mode(self.last_plot_mode)

    def _render_plot_mode(self, mode: str | None) -> None:
        if mode in ("engineering_phase", "engineering_polarization", "engineering_measure"):
            panel = getattr(self, "inline_analysis", None)
            if panel is not None:
                panel.render_plot_mode(mode.removeprefix("engineering_"))
            return
        renderer = {
            "azimuth_rect": self._plot_azimuth_rect,
            "azimuth_polar": self._plot_azimuth_polar,
            "frequency": self._plot_frequency,
            "elevation_sweep": self._plot_elevation_sweep,
            "waterfall": self._plot_waterfall,
            "isar_image": self._plot_isar_image,
            "az_vs_range": self._plot_az_vs_range,
            "compare": self._plot_compare,
            "delta_map": self._plot_delta_map,
            "cdf": self._plot_cdf,
            "sector_stats": self._plot_sector_stats,
            "range_freq": self._plot_range_freq,
        }.get(mode)
        if renderer is not None:
            renderer()

    def _on_analysis_setting_changed(self, kind: str) -> None:
        """Re-render when an analysis setting affects this plot."""
        save_preferences = getattr(self, "_save_analysis_preferences", None)
        if callable(save_preferences):
            save_preferences()
        mode = self.last_plot_mode
        if kind == "heatmap_position":
            # A cursor move changes only the frozen display recipe. Keep its
            # original dataset selections even if the lists have since changed.
            spec = getattr(self, "last_python_plot_spec", None)
            if spec and spec[0] == "supported" and spec[3] == "waterfall":
                parameters = dict(spec[4])
                options = dict(parameters.get("heatmap_options", {}))
                options["cut_position"] = self.image_analysis_controls.options()["cut_position"]
                parameters["heatmap_options"] = options
                self.last_python_plot_spec = (*spec[:4], parameters)
            return
        affected = {
            "pbp": mode in ("azimuth_rect", "azimuth_polar", "frequency")
            and self._button_checked(getattr(self, "btn_pbp", None)),
            "cdf": mode == "cdf",
            "sector": mode == "sector_stats",
            "range": mode == "range_freq",
            "sweep": mode in ("frequency", "elevation_sweep"),
            "heatmap": mode == "waterfall",
        }.get(kind, False)
        if not affected or getattr(self, "_active_plot_tab", "plotting") != "plotting":
            return
        if mode != "waterfall" and self._button_checked(getattr(self, "btn_hold", None)):
            self.status.showMessage(
                "Setting saved. Hold is on, so plot again to apply it to the held canvas."
            )
            return
        if mode == 'cdf':
            from GRIM_Backend.plotting.modes.cdf_mode import update_threshold
            if update_threshold(self):
                self._capture_successful_python_plot('cdf')
                return
        self._render_plot_mode(mode)

    def _maybe_autoscale(self) -> None:
        """Auto-fit the view after a render when the Auto Scale toggle is on.

        Mirrors a Fit Both click so axes track the current data without the
        user reaching for the button after every (re)plot.
        """
        if self._button_checked(getattr(self, "btn_auto_scale", None)) and self.last_plot_mode is not None:
            self._fit_both()
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None:
            overlays.refresh()

    def _on_auto_scale_toggled(self) -> None:
        # Apply at once so enabling the toggle fits whatever is already plotted.
        self._maybe_autoscale()

    def _on_pbp_toggled(self) -> None:
        if self.last_plot_mode is None:
            return
        # As with Auto Plot, a toggle must not append every selected series to
        # a held canvas; the next explicit plot adds the band or the curves.
        if self._button_checked(getattr(self, "btn_hold", None)):
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()

    def _on_waterfall_style_changed(self) -> None:
        if self.last_plot_mode == "delta_map":
            self._plot_delta_map()
            return
        if self.last_plot_mode == "range_freq":
            self._plot_range_freq()
            return
        if self.last_plot_mode not in ("waterfall", "isar_image", "az_vs_range"):
            return
        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()
        else:
            self._plot_isar_image()

    def _on_colormap_changed(self) -> None:
        if self.last_plot_mode == "delta_map":
            self._update_current_python_plot_style()
            for mesh in self.plot_ax.collections:
                if hasattr(mesh, "_grim_delta_labels"):
                    delta_map_mode.set_colormap(mesh, self._effective_colormap())
            self.plot_canvas.draw_idle()
            return
        # A colormap switch is a style-only change. Reuse the ScalarMappables
        # already on the canvas instead of repeating FFTs, gridding, or large
        # dataset selections. Fall back to a render only when no live mapped
        # artist exists (for example, before the first image was drawn).
        cmap = self._effective_colormap()
        self._update_current_python_plot_style()
        updated = False
        seen: set[int] = set()
        for ax in self.plot_figure.axes:
            for artist in [*ax.images, *ax.collections]:
                if id(artist) in seen or not hasattr(artist, "set_cmap"):
                    continue
                seen.add(id(artist))
                try:
                    values = artist.get_array()
                except Exception:
                    values = None
                if values is None:
                    continue
                artist.set_cmap(cmap)
                updated = True
        for colorbar in self.plot_colorbars:
            mappable = getattr(colorbar, "mappable", None)
            if mappable is not None and hasattr(mappable, "set_cmap"):
                mappable.set_cmap(cmap)
                try:
                    colorbar.update_normal(mappable)
                except Exception:
                    pass
                updated = True
        if updated:
            self.plot_canvas.draw_idle()
            return

        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()
        elif self.pbp_fill_mode in ("heatmap_rcs", "heatmap_density"):
            if self.last_plot_mode == "azimuth_rect":
                self._plot_azimuth_rect()
            elif self.last_plot_mode == "azimuth_polar":
                self._plot_azimuth_polar()
            elif self.last_plot_mode == "frequency":
                self._plot_frequency()

    @staticmethod
    def _reset_plot_color_limits(context) -> None:
        """Discard limits expressed in the previous display scale for this tab.

        Equal endpoints and a zero tick step select automatic color scaling.
        Block live callbacks so an ISAR scale edit still waits for Apply and
        the old image is not temporarily recolored with the new scale's limits.
        """
        for spin in (context.spin_plot_zmin, context.spin_plot_zmax,
                     context.spin_plot_zstep):
            was_blocked = spin.blockSignals(True)
            try:
                spin.setValue(0.0)
            finally:
                spin.blockSignals(was_blocked)

    def _on_plot_scale_changed(self) -> None:
        if self.last_plot_mode is None:
            self._apply_plot_theme()
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
            self._fit_y()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
            self._fit_y()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
            if getattr(self, "_ordinary_plot_pending", False):
                self._ordinary_plot_fit = "_fit_y"
            else:
                self._fit_y()
        elif self.last_plot_mode == "elevation_sweep":
            self._plot_elevation_sweep()
            if getattr(self, "_ordinary_plot_pending", False):
                self._ordinary_plot_fit = "_fit_y"
            else:
                self._fit_y()
        elif self.last_plot_mode == "compare":
            self._plot_compare()
        elif self.last_plot_mode == "cdf":
            self._plot_cdf()
            if getattr(self, "_ordinary_plot_pending", False):
                self._ordinary_plot_fit = "_fit_both"
            else:
                self._fit_both()
        elif self.last_plot_mode == "sector_stats":
            self._plot_sector_stats()
            self._fit_y()
        elif self.last_plot_mode == "range_freq":
            self._plot_range_freq()
        elif self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()

    def _plot_scale_mode(self) -> str:
        scale = self.combo_plot_scale.currentData()
        if scale in ("dbsm", "linear"):
            return scale
        return "dbsm"

    def _plot_scale_is_linear(self) -> bool:
        return self._plot_scale_mode() == "linear"

    @staticmethod
    def _phase_wrap_mode(dataset: RcsGrid) -> str:
        mode = str((dataset.units or {}).get("phase_wrap", "-180_180")).strip()
        return "0_360" if mode == "0_360" else "-180_180"

    def _wrap_phase_degrees(self, dataset: RcsGrid, values):
        phase = np.asarray(values, dtype=float)
        finite = np.isfinite(phase)
        wrapped = np.full(phase.shape, np.nan, dtype=float)
        if self._phase_wrap_mode(dataset) == "0_360":
            wrapped[finite] = np.mod(phase[finite], 360.0)
        else:
            wrapped[finite] = np.mod(phase[finite] + 180.0, 360.0) - 180.0
        return wrapped

    def _phase_display_degrees(self, dataset: RcsGrid, values):
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            valid = np.isfinite(raw) & (np.abs(raw) > 0.)
            phase_radians = np.where(valid, np.angle(raw), np.nan)
        else:
            phase_radians = raw.astype(float)
        return self._wrap_phase_degrees(dataset, np.degrees(phase_radians))

    def _rcs_display_values(self, dataset: RcsGrid, rcs_values, frequency_value=None):
        if self._button_checked(self.btn_phase):
            return self._phase_display_degrees(dataset, rcs_values)
        if self._plot_scale_is_linear():
            return dataset.rcs_to_linear(rcs_values)
        return self._display_from_linear(dataset, dataset.rcs_to_linear(rcs_values), frequency_value)

    def _rcs_axis_label(self) -> str:
        if self._button_checked(self.btn_phase):
            return "Phase (deg)"
        dataset = getattr(self, "active_dataset", None)
        quantity = dataset.linear_quantity() if isinstance(dataset, RcsGrid) else "sigma_3d"
        quantity_name, linear_unit = self._linear_quantity_label_and_unit(quantity)
        if self._plot_scale_is_linear():
            return f"{quantity_name} ({linear_unit})"
        unit = dataset.default_log_unit() if isinstance(dataset, RcsGrid) else "dBsm"
        return f"{quantity_name} ({unit})"


    def _polar_zero_location(self) -> str:
        loc = self.combo_polar_zero.currentData()
        if isinstance(loc, str) and loc:
            return loc
        return "N"

    def _apply_polar_orientation(self, ax) -> None:
        if ax.name != "polar":
            return
        ax.set_theta_zero_location(self._polar_zero_location())
        # Compass convention: azimuth increases clockwise.
        ax.set_theta_direction(-1)
        # Label tick marks in (-180, 180] so -90 shows on the left (W) under
        # the default N-up/CW orientation, matching the compass grid.
        self._set_polar_thetagrids(ax, np.arange(0.0, 360.0, 30.0))

    def _set_polar_thetagrids(self, ax, tick_degrees) -> None:
        tick_degrees = np.asarray(tick_degrees, dtype=float)
        signed_degrees = np.where(tick_degrees <= 180.0, tick_degrees, tick_degrees - 360.0)
        display_unit = str(getattr(self, "_polar_display_unit", "deg"))
        labels_values = plot_common.convert_axis_values(
            signed_degrees, "azimuth", "deg", display_unit
        )
        if display_unit == "rad":
            labels = [f"{value:.3g}" for value in labels_values]
        else:
            labels = [f"{value:g}" for value in labels_values]
        ax.set_thetagrids(tick_degrees, labels=labels)

    def _apply_polar_zero_direction(self) -> None:
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._apply_polar_orientation(ax)

    def _on_polar_zero_changed(self) -> None:
        self._apply_polar_zero_direction()
        self.plot_canvas.draw_idle()

    def _edges_from_centers(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            step = 1.0
            return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)
        diffs = np.diff(values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = values[:-1] + diffs / 2.0
        edges[0] = values[0] - diffs[0] / 2.0
        edges[-1] = values[-1] + diffs[-1] / 2.0
        return edges

    def _plot_pbp_heatmap(
        self,
        x_values,
        y_min,
        y_max,
        *,
        density: np.ndarray | None = None,
        key=PBP_BAND_KEY,
    ) -> None:
        x_values = np.asarray(x_values, dtype=float)
        y_min = np.asarray(y_min, dtype=float)
        y_max = np.asarray(y_max, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_min) & np.isfinite(y_max)
        if not np.any(valid):
            return

        coverage_bar = None
        coverage_max = max(1., float(np.nanmax(density))) if density is not None and len(density) else 1.

        def draw_segment(seg_x, seg_min, seg_max, seg_density=None) -> None:
            nonlocal coverage_bar
            lower = np.minimum(seg_min, seg_max)
            upper = np.maximum(seg_min, seg_max)
            if seg_x.size == 0:
                return

            x_edges = self._edges_from_centers(seg_x)
            lower_edges = np.interp(x_edges, seg_x, lower, left=lower[0], right=lower[-1])
            upper_edges = np.interp(x_edges, seg_x, upper, left=upper[0], right=upper[-1])

            samples = max(8, int(self.pbp_heatmap_samples))
            y_edges = np.vstack(
                [np.linspace(lo, hi, samples + 1) for lo, hi in zip(lower_edges, upper_edges)]
            ).T
            if self.pbp_fill_mode == "heatmap_density":
                if seg_density is None:
                    return
                values = np.tile(seg_density, (samples, 1))
            else:
                values = np.vstack(
                    [np.linspace(lo, hi, samples) for lo, hi in zip(lower, upper)]
                ).T
            x_grid = np.tile(x_edges, (samples + 1, 1))

            cmap = self._effective_colormap()
            coverage = self.pbp_fill_mode == "heatmap_density"
            limits = {"vmin": 0., "vmax": coverage_max} if coverage else {}
            mesh = self.plot_ax.pcolormesh(x_grid, y_edges, values, shading="auto", cmap=cmap, **limits)
            mesh._grim_dataset_key = key
            if coverage and coverage_bar is None:
                from matplotlib.ticker import MaxNLocator
                coverage_bar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
                coverage_bar._grim_dataset_key = key
                coverage_bar.set_label("Valid cuts (count at each coordinate)")
                coverage_bar.locator = MaxNLocator(integer=True)
                coverage_bar.update_ticks()
                self.plot_colorbars.append(coverage_bar)
                self._note_plot_render("PBP fill color shows valid-cut coverage, not return-level density.")

        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                seg = slice(start, idx)
                seg_density = None
                if density is not None:
                    seg_density = np.asarray(density, dtype=float)[seg]
                draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)
                start = None
        if start is not None:
            seg = slice(start, len(valid))
            seg_density = None
            if density is not None:
                seg_density = np.asarray(density, dtype=float)[seg]
            draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)

    def _plot_pbp_fill(
        self,
        x_values,
        y_min,
        y_max,
        label: str,
        polar: bool,
        *,
        density: np.ndarray | None = None,
        key=PBP_BAND_KEY,
        color: str | None = None,
        heatmap: bool = False,
    ) -> None:
        if heatmap:
            self._plot_pbp_heatmap(x_values, y_min, y_max, density=density, key=key)
            (proxy,) = self.plot_ax.plot([], [], color=color, label=label)
            self._register_plot_line(proxy, key)
            return
        fill = self.plot_ax.fill_between(
            x_values,
            y_min,
            y_max,
            color=color,
            alpha=0.35,
            linewidth=0,
            label=label,
        )
        fill._grim_dataset_key = key

    def _plot_pbp_band(
        self,
        x_values,
        lower,
        upper,
        label: str,
        polar: bool,
        *,
        density: np.ndarray | None = None,
        key=PBP_BAND_KEY,
        color: str | None = None,
        heatmap: bool = False,
        native_edges=None,
    ) -> None:
        """Append one PBP fill and its edges as one selectable plot item."""
        if color is None:
            color = self.plot_ax._get_lines.get_next_color()
        self._plot_pbp_fill(
            x_values, lower, upper, label, polar, density=density, key=key,
            color=color, heatmap=heatmap,
        )
        for edge_index, edge in enumerate((lower, upper)):
            for line in self._plot_bounded_line(
                self.plot_ax, x_values, edge, color=color, linewidth=1,
                label="_nolegend_", zorder=1.9,
                native_values=(native_edges[0], native_edges[edge_index + 1]) if native_edges is not None else None,
            ):
                self._register_plot_line(line, key)

    def _new_pbp_bands(self, datasets) -> _PbpBands:
        return _PbpBands(self, datasets)

    def _pbp_percentiles(self) -> tuple[float, float] | None:
        controls = getattr(self, "analysis_controls", None)
        return controls.pbp_percentiles() if controls is not None else None

    def _style_axes(self, ax) -> None:
        bg = self._current_plot_bg()
        grid = self._current_plot_grid()
        text = self._current_plot_text()
        ax.set_facecolor(bg)
        grid_on = self._plot_grid_enabled()
        ax.grid(grid_on, color=grid, alpha=0.35)
        ax.tick_params(colors=text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        for annotation in ax.texts:
            if annotation.get_label() == "_iq_guide":
                annotation.set_color(text)
        if hasattr(ax, "zaxis") and ax.zaxis is not None:
            ax.zaxis.label.set_color(text)
        if hasattr(ax, "spines"):
            for spine in ax.spines.values():
                spine.set_color(self.application_palette["border"])
        if ax.name == "polar":
            self._apply_polar_orientation(ax)

    def _style_plot_axes(self) -> None:
        self.plot_figure.set_facecolor(self._current_plot_bg())
        self._style_axes(self.plot_ax)

    def _plot_grid_enabled(self) -> bool:
        checkbox = getattr(self, "chk_plot_grid_visible", None)
        if checkbox is None:
            return True
        return bool(checkbox.isChecked())

    def _current_plot_bg(self) -> str:
        return self.plot_bg_color or self.application_palette["panel_bg"]

    def _current_plot_grid(self) -> str:
        return self.plot_grid_color or self.application_palette["grid"]

    def _current_plot_text(self) -> str:
        return self.plot_text_color or self.application_palette["text"]

    def _apply_plot_theme(self) -> None:
        self._update_current_python_plot_style()
        self.plot_figure.set_facecolor(self._current_plot_bg())
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._style_axes(ax)
            legend = ax.get_legend()
            if legend is not None:
                self._configure_legend(legend, ax)
                for text in legend.get_texts():
                    text.set_color(self._current_plot_text())
                legend.get_frame().set_facecolor(self._current_plot_bg())
                legend.get_frame().set_edgecolor(self._current_plot_grid())
                self._sync_dataset_legend(ax)
        for colorbar in self.plot_colorbars:
            label_text = colorbar.ax.get_ylabel() or self._rcs_axis_label()
            colorbar.set_label(label_text, color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())
        self.plot_canvas.draw_idle()

    def _apply_colorbar_ticks(self, colorbar) -> None:
        zstep = self.spin_plot_zstep.value()
        if zstep <= 0.0:
            return
        try:
            vmin, vmax = colorbar.mappable.get_clim()
        except Exception:
            return
        if vmin is None or vmax is None:
            return
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        ticks = plot_common.bounded_ticks(vmin, vmax, zstep)
        if ticks is None:
            ticks = np.linspace(vmin, vmax, plot_common.MAX_EXPLICIT_TICKS)
            self._note_plot_render(
                f"Colorbar tick step requested over {plot_common.MAX_EXPLICIT_TICKS} ticks; "
                "increase the Z tick step."
            )
        colorbar.set_ticks(ticks)

    def _choose_plot_color(self, which: str) -> None:
        if which == "bg":
            current = self._current_plot_bg()
            title = "Select Plot Background Color"
        elif which == "grid":
            current = self._current_plot_grid()
            title = "Select Plot Grid Color"
        else:
            current = self._current_plot_text()
            title = "Select Plot Text Color"
        color = QColorDialog.getColor(QColor(current), self, title)
        if not color.isValid():
            return
        if which == "bg":
            self.plot_bg_color = color.name()
        elif which == "grid":
            self.plot_grid_color = color.name()
        else:
            self.plot_text_color = color.name()
        self._update_plot_color_buttons()
        self._apply_plot_theme()

    def _update_plot_color_buttons(self) -> None:
        self.btn_plot_bg.setStyleSheet(f"background: {self._current_plot_bg()};")
        self.btn_plot_grid.setStyleSheet(f"background: {self._current_plot_grid()};")
        self.btn_plot_text.setStyleSheet(f"background: {self._current_plot_text()};")

    def _remove_colorbar(self) -> None:
        if not self.plot_colorbars:
            return
        for colorbar in self.plot_colorbars:
            try:
                if colorbar.ax is not None:
                    colorbar.remove()
            except Exception:
                pass
        self.plot_colorbars = []

    def _ensure_axes(self, projection: str) -> None:
        desired = "polar" if projection == "polar" else "rectilinear"
        # Axes.clear() retains equal aspect, anchors, and colorbar geometry.
        # A single I/Q panel therefore needs new axes just like a multi-panel diagnostic.
        if (self.plot_ax.name == desired and self.plot_axes is None
                and not hasattr(self.plot_ax, "_grim_delta_map")
                and not getattr(self.plot_ax, "_grim_engineering_axis", False)):
            return
        self._release_native_plot_samples()
        self._remove_colorbar()
        self.plot_figure.clear()
        if desired == "polar":
            self.plot_ax = self.plot_figure.add_subplot(111, projection="polar")
        else:
            self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self._style_plot_axes()

    def _clear_plot(self) -> None:
        from GRIM_Backend.plotting.modes.cdf_mode import clear_state
        clear_state(self.plot_figure)
        context = getattr(self, "_plot_contexts", {}).get(getattr(self, "_active_plot_tab", None))
        if context is not None:
            context.session.disposed = False
            context.session.cancel()
        panel = getattr(self, "inline_analysis", None)
        if panel is not None:
            panel.deactivate()
        self.plot_figure._grim_independent_y = False
        self.plot_figure._grim_independent_x = False
        self.plot_figure._grim_engineering_signature = None
        self.plot_figure._grim_iq_held = None
        # Clear starts a blank canvas independent of the previous plot's layout.
        self.plot_figure.set_layout_engine(None)
        self.plot_figure._grim_engineering_layout = False
        self._release_native_plot_samples()
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None:
            overlays.clear()
        if getattr(self, "_active_plot_tab", "plotting") == "plotting":
            # Markers live on the Plotting canvas only.
            self._clear_plot_markers(redraw=False)
        self._set_compare_sector_controls_visible(False)
        controls = getattr(self, "delta_map_controls", None)
        if controls is not None:
            controls.hide()
        self._plot_render_generation = (
            int(getattr(self, "_plot_render_generation", 0)) + 1
        )
        self._remove_colorbar()
        self.plot_figure.clear()
        self.plot_figure._grim_line_plot_signature = None
        self.plot_figure._grim_held_phase_datasets = []
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self._style_plot_axes()
        self._apply_plot_limits()
        if getattr(self, "_active_plot_tab", None) == "isar":
            # Never let Export Plot treat the newly blank canvas as the last
            # completed ISAR figure.
            self._invalidate_isar_figure()
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None:
            recorder.invalidate_current_plot()
        self.last_python_plot_spec = None

        if overlays is not None:
            overlays.refresh()

    def _single_selection_index(self, widget: QListWidget, label: str) -> int | None:
        selected = sorted(self._selected_indices(widget))
        if len(selected) != 1:
            count = len(selected)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return selected[0]

    def _single_selection_value(self, widget: QListWidget, label: str):
        values = self._selected_values(widget)
        if len(values) != 1:
            count = len(values)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return values[0]

    def _overlay_polarizations(self):
        values = tuple(dict.fromkeys(str(value) for value in self._selected_values(self.list_pol)))
        if not values:
            self.status.showMessage("Select one or more polarizations to plot.")
            return None
        return values

    def _native_sample_overlay_enabled(self) -> bool:
        mode = getattr(self, "last_plot_mode", None)
        return (
            mode in NATIVE_DB_LINE_MODES
            and not (mode in ("azimuth_rect", "azimuth_polar", "frequency")
                     and self._button_checked(self.btn_pbp))
        )

    def _overlay_axis_selection(self, reference, dataset, axis: str, values):
        if not self._native_sample_overlay_enabled():
            return self._axis_selection_for_dataset(reference, dataset, axis, values)
        try:
            return plot_common.native_axis_selection(reference, dataset, axis, values)
        except ValueError as exc:
            self._plot_selection_failed = True
            self._note_plot_render(f"Incompatible {axis} selection: {exc}.")
            return None

    @staticmethod
    def _button_checked(button: QToolButton | None) -> bool:
        return bool(button.isChecked()) if button is not None else False

    # --- shared display-quantity helpers (used by the plot_modes modules) ----

    def _display_from_values(self, dataset, values, frequency_value, *, preserve_zeros=False):
        """Raw samples -> the current display quantity: phase (deg) when the
        Phase toggle is on, otherwise linear or the dataset's default dB per
        the plot-scale combo. `frequency_value` may be scalar or an array
        broadcastable against `values` (frequency-dependent dB conversions)."""
        if self._button_checked(self.btn_phase):
            return self._phase_display_degrees(dataset, values)
        linear = dataset.rcs_to_linear(values)
        if self._plot_scale_is_linear():
            return linear
        return self._display_from_linear(dataset, linear, frequency_value, preserve_zeros=preserve_zeros)

    def _display_from_linear(self, dataset, linear_values, frequency_value, *, preserve_zeros=False):
        """Already-linear power values -> current display scale (no phase branch;
        used by the P50/statistics paths that aggregate in linear space)."""
        if self._plot_scale_is_linear():
            return linear_values
        linear_values = np.asarray(linear_values, dtype=float)
        positive = np.isfinite(linear_values) & (linear_values > 0.0)
        if not preserve_zeros and np.any(np.isfinite(linear_values) & ~positive):
            self._note_plot_render(
                "Zero/nonpositive samples are omitted on logarithmic axes; "
                "positive levels are not floored."
            )
        return dataset.linear_to_default_db(
            np.where(positive | (preserve_zeros & (linear_values == 0.)), linear_values, np.nan),
            frequency_value=frequency_value, eps=0.0,
        )

    def _display_unit(self, datasets) -> str:
        """Unit string for the current display quantity."""
        if self._button_checked(self.btn_phase):
            return "deg"
        if self._plot_scale_is_linear():
            quantity = datasets[0][1].linear_quantity()
            return self._linear_quantity_label_and_unit(quantity)[1]
        quantities = {self._native_db_quantity(ds) for _, ds in datasets}
        return next(iter(quantities))[1] if len(quantities) == 1 else "Mixed dB"

    @staticmethod
    def _native_db_quantity(dataset) -> tuple[str, str]:
        return (str(dataset.linear_quantity()).strip().lower(), str(dataset.default_log_unit()))

    def _native_db_overlay_enabled(self) -> bool:
        return (
            getattr(self, "last_plot_mode", None) in (*NATIVE_DB_LINE_MODES, "waterfall")
            and not self._button_checked(self.btn_phase)
            and not self._plot_scale_is_linear()
        )

    @staticmethod
    def _tag_native_db_artist(artist, quantities) -> None:
        if quantities:
            artist._grim_db_quantities = frozenset(quantities)
            artist._grim_db_base_label = artist.get_label()

    def _refresh_native_db_labels(self, ax) -> None:
        """Label the actual displayed curves, including retained Hold curves."""
        value_axis = getattr(ax, "_grim_db_value_axis", None)
        if value_axis is None:
            return
        artists = [
            artist for artist in (*ax.lines, *ax.collections)
            if getattr(artist, "_grim_db_quantities", None)
        ]
        quantities = {entry for artist in artists for entry in artist._grim_db_quantities}
        if not quantities:
            return
        mixed = len(quantities) > 1
        if mixed:
            axis_label = "Mixed dB"
        else:
            quantity, unit = next(iter(quantities))
            name, _ = self._linear_quantity_label_and_unit(quantity)
            tag = getattr(ax, "_grim_db_label_tag", "")
            axis_label = f"{name}{tag} ({unit})"
        (ax.set_xlabel if value_axis == "x" else ax.set_ylabel)(axis_label)
        for artist in artists:
            label = artist._grim_db_base_label
            if mixed and isinstance(label, str) and label and not label.startswith("_"):
                units = ", ".join(sorted({entry[1] for entry in artist._grim_db_quantities}))
                head, separator, tail = label.partition(" | ")
                label = f"{head} [{units}]{separator}{tail}"
            artist.set_label(label)

    @staticmethod
    def _linear_quantity_label_and_unit(quantity) -> tuple[str, str]:
        """Return the physical name and SI unit of stored linear power."""

        key = str(quantity).strip().lower()
        return {
            "sigma_3d": ("RCS", "m²"),
            "sigma_2d": ("Scattering Width", "m"),
            "power_ratio": ("Power Ratio", "dimensionless"),
            "ratio": ("Power Ratio", "dimensionless"),
        }.get(key, ("Value", "linear"))

    def _display_axis_label(self, datasets, tag: str = "") -> str:
        """Axis/colorbar label for the current display quantity; `tag` is an
        optional qualifier inserted after the quantity name (e.g. " P50")."""
        if self._button_checked(self.btn_phase):
            return f"Phase{tag} (deg)"
        quantity = str(datasets[0][1].linear_quantity()).strip().lower()
        quantity_name, linear_unit = self._linear_quantity_label_and_unit(quantity)
        if self._plot_scale_is_linear():
            return f"{quantity_name}{tag} ({linear_unit})"
        if getattr(self, "last_plot_mode", None) in NATIVE_DB_LINE_MODES:
            self.plot_ax._grim_db_label_tag = tag
        if self._display_unit(datasets) == "Mixed dB":
            return "Mixed dB"
        return f"{quantity_name}{tag} ({self._display_unit(datasets)})"

    # --- renderer preflight, unit conversion, and display bounding ----------

    def _start_plot_render(self) -> None:
        from GRIM_Backend.plotting.modes.cdf_mode import clear_state
        if self.last_plot_mode != 'cdf':
            clear_state(self.plot_figure)
        context = getattr(self, "_plot_contexts", {}).get(getattr(self, "_active_plot_tab", None))
        if context is not None:
            context.session.disposed = False
            context.session.cancel()
        panel = getattr(self, "inline_analysis", None)
        if panel is not None and not panel._rendering:
            panel.deactivate()
        if not str(self.last_plot_mode).startswith("engineering_"):
            if getattr(self.plot_figure, "_grim_engineering_layout", False):
                self.plot_figure.set_layout_engine(None)
                self.plot_figure._grim_engineering_layout = False
            self.plot_figure._grim_independent_y = False
            self.plot_figure._grim_independent_x = False
            self.plot_figure._grim_engineering_signature = None
            self.plot_figure._grim_iq_held = None
        else:
            self._set_compare_sector_controls_visible(False)
        image_controls = getattr(self, "image_analysis_controls", None)
        if image_controls is not None and self.last_plot_mode != "waterfall":
            bar = getattr(image_controls, "cut_bar", None)
            if bar is not None:
                bar.hide()
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None:
            overlays.stop_interaction()
        controls = getattr(self, "delta_map_controls", None)
        if controls is not None and self.last_plot_mode != "delta_map":
            controls.hide()
        if self.last_plot_mode != "delta_map" and getattr(self.plot_figure, "_grim_delta_layout", False):
            self.plot_figure.set_layout_engine(None)
            self.plot_figure._grim_delta_layout = False
        if self.last_plot_mode != 'isar_image' and getattr(self.plot_figure, '_grim_isar_layout', False):
            self.plot_figure.set_layout_engine(None)
            self.plot_figure._grim_isar_layout = False
        if self.last_plot_mode not in ("frequency", "elevation_sweep", "cdf") and getattr(
            self.plot_figure, "_grim_analysis_constrained_layout", False
        ):
            self.plot_figure.set_layout_engine(None)
            self.plot_figure._grim_analysis_constrained_layout = False
        selection_notice = getattr(self, "_pending_parameter_selection_notice", None)
        self._plot_render_notes = [selection_notice] if selection_notice else []
        self._pending_parameter_selection_notice = None
        self._plot_line_point_limit = plot_common.MAX_LINE_POINTS
        self._plot_selection_failed = False
        self._plot_render_generation = (
            int(getattr(self, "_plot_render_generation", 0)) + 1
        )

    def _note_plot_render(self, note: str) -> None:
        notes = getattr(self, "_plot_render_notes", None)
        if notes is None:
            notes = self._plot_render_notes = []
        if note and note not in notes:
            notes.append(note)

    def _show_plot_status(self, message: str) -> None:
        self._refresh_display_reduction_notice()
        notes = list(getattr(self, "_plot_render_notes", []))
        if notes:
            message = f"{message} " + " ".join(notes)
        self.status.showMessage(message)

    @staticmethod
    def _release_artist_native_samples(artist):
        native = getattr(artist, "_grim_native_samples", None)
        if native is not None:
            # Image records may share a view of the native mapping. Captures
            # own separate frozen storage; live records expire with the plot.
            artist._grim_image_record = None
            artist._grim_native_rectilinear_data = None
            native.close()
            artist._grim_native_samples = None
        artist._grim_native_rectilinear_data = None

    def _release_native_plot_samples(self):
        context = getattr(self, "_plot_contexts", {}).get(getattr(self, "_active_plot_tab", None))
        if context is not None:
            context.session.release()
        linked_cuts = getattr(self.plot_figure, "_grim_linked_cuts", None)
        if linked_cuts is not None:
            linked_cuts.disconnect()
            self.plot_figure._grim_linked_cuts = None
        for ax in self.plot_figure.axes:
            for artist in (*ax.lines, *ax.collections, *ax.images):
                self._release_artist_native_samples(artist)

    def _dispose_native_plot_snapshots(self):
        """Release both tabs' temporary resources after an accepted close."""
        for name in ("_autoplot_timer", "_hover_timer", "_plot_slider_refresh_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()
        for context in getattr(self, "_plot_contexts", {}).values():
            context.session.close()
            if context.plot_slider is not None:
                context.plot_slider.stop_interaction()
            for marker in context.session.state.pop("_plot_markers", []):
                self._detach_marker(marker)
            context.session.state.update(_active_marker=None, _marker_drag=None)
            panel = getattr(context, "inline_analysis", None)
            if panel is not None:
                panel.deactivate()
        figures = {context.plot_figure for context in getattr(self, "_plot_contexts", {}).values()}
        figures.add(self.plot_figure)
        self._clear_plot_markers(redraw=False)
        for figure in figures:
            cdf_mode.clear_state(figure)
            figure._grim_iq_held = None
            linked_cuts = getattr(figure, "_grim_linked_cuts", None)
            if linked_cuts is not None:
                linked_cuts.disconnect()
                figure._grim_linked_cuts = None
            for ax in figure.axes:
                for artist in (*ax.lines, *ax.collections, *ax.images):
                    if getattr(artist, "_grim_native_samples", None) is not None:
                        self._release_artist_native_samples(artist)
                        # A closed Qt window can be shown again by its owner.
                        # Its existing reduced view then has display-only
                        # probes until the next plot rebuilds the snapshot.
                        artist._grim_native_probe_unavailable = True
            self._refresh_display_reduction_notice(figure)

    def _refresh_display_reduction_notice(self, figure=None):
        figure = self.plot_figure if figure is None else figure
        old = getattr(figure, "_grim_reduction_notice", None)
        if old is not None and old in figure.texts:
            old.remove()
        figure._grim_reduction_notice = None
        reduced = [artist for ax in figure.axes
                   for artist in (*ax.lines, *ax.collections, *ax.images)
                   if getattr(artist, "_grim_display_reduced", False)]
        if not reduced:
            return
        fallback = any(getattr(a, "_grim_native_probe_unavailable", False) for a in reduced)
        overview = any(getattr(a, "_grim_native_probe", None) is not None for a in reduced)
        suffix = "display-only probing (native snapshot unavailable)" if fallback else (
            "native samples / labeled block-peak probes" if overview else "probes use native samples")
        figure._grim_reduction_notice = figure.text(
            0.995, 0.005, "Display reduced; " + suffix, ha="right", va="bottom",
            fontsize=8, color=self._current_plot_text(),
            bbox={"facecolor": self._current_plot_bg(), "alpha": 0.9, "edgecolor": "none"},
        )

    def _preflight_plot_datasets(self, datasets):
        try:
            plot_common.validate_plot_datasets(
                datasets,
                phase=self._button_checked(self.btn_phase),
                linear=self._plot_scale_is_linear(),
                allow_mixed_db=self._native_db_overlay_enabled(),
            )
        except ValueError as exc:
            self.status.showMessage(f"Plot blocked: {exc}.")
            return None
        if self._button_checked(self.btn_phase):
            for note in plot_common.coherent_metadata_plot_warnings(datasets):
                self._note_plot_render("Phase provenance note: " + note + ".")
        return plot_common.reference_dataset(datasets, self.active_dataset)

    def _line_plot_signature(self, mode: str, projection: str, reference, datasets):
        """Describe the coordinate/ordinate contract of a line-plot canvas."""

        phase = self._button_checked(self.btn_phase)
        if phase:
            # Hold compares what is drawn on the axes. Phase provenance is
            # displayed as a warning, not treated as an ordinate incompatibility.
            ordinate = ("phase", "deg")
        elif self._native_db_overlay_enabled():
            ordinate = ("native_logarithmic", "dB")
        else:
            quantity = str(datasets[0][1].linear_quantity()).strip().lower()
            display_unit = (
                self._linear_quantity_label_and_unit(quantity)[1]
                if self._plot_scale_is_linear()
                else str(datasets[0][1].default_log_unit())
            )
            ordinate = (quantity, display_unit)
        if mode == "cdf":
            controls = getattr(self, "analysis_controls", None)
            exceedance = bool(controls is not None and controls.cdf_exceedance())
            return ("cdf", str(projection), ordinate, exceedance)
        # Sector statistics share the azimuth axis and level ordinate of an
        # Azimuth (Rect) cut, so Hold can lay sector levels over the cut.
        signature_mode = "azimuth_rect" if mode == "sector_stats" else str(mode)
        return (
            signature_mode,
            str(projection),
            self._plot_axis_unit(reference, {
                "azimuth_rect": "azimuth",
                "azimuth_polar": "azimuth",
                "frequency": "frequency",
                "elevation_sweep": "elevation",
                "sector_stats": "azimuth",
            }[mode]),
            ordinate,
        )

    def _prepare_line_plot_axes(
        self,
        mode: str,
        projection: str,
        reference,
        datasets,
    ) -> bool:
        """Clear or validate a line canvas before honoring the Hold toggle.

        Hold is an overlay operation, so it may retain only a line plot (and
        its PBP band) with the same x-coordinate and ordinate contract.
        Images, multi-panel layouts, phase/scale changes, and unlike linear
        quantities require a fresh canvas. Native dB overlays share a numeric
        scale and identify mixed units on the axis and individual curves.
        """

        hold = self._button_checked(self.btn_hold)
        signature = self._line_plot_signature(mode, projection, reference, datasets)
        desired_projection = "polar" if projection == "polar" else "rectilinear"
        figure = self.plot_figure
        band_colorbar_axes = {
            bar.ax for bar in self.plot_colorbars
            if is_pbp_band_key(getattr(bar, "_grim_dataset_key", None))
        }
        axes = [ax for ax in figure.axes if ax not in band_colorbar_axes]
        has_images_or_collections = any(
            not is_pbp_band_key(getattr(artist, "_grim_dataset_key", None))
            for ax in axes
            for artist in (*ax.images, *ax.collections)
        )
        has_content = any(ax.images or ax.collections or ax.lines for ax in axes)
        prior_signature = getattr(figure, "_grim_line_plot_signature", None)

        if hold and has_content:
            compatible_layout = (
                self.plot_axes is None
                and getattr(self.plot_ax, "name", "") == desired_projection
                and not has_images_or_collections
            )
            if not compatible_layout or prior_signature != signature:
                self.status.showMessage(
                    "Hold blocked: the existing canvas uses a different plot axis, "
                    "coordinate/unit system, scale, or physical quantity. Turn off "
                    "Hold or Clear the plot before continuing."
                )
                return False

        self._ensure_axes(projection)
        if not hold:
            self._release_native_plot_samples()
            self._remove_colorbar()
            self.plot_ax.clear()
            self._style_plot_axes()
        figure._grim_line_plot_signature = signature
        self.plot_ax._grim_db_value_axis = (
            ("x" if mode == "cdf" else "y")
            if self._native_db_overlay_enabled() else None
        )
        self.plot_ax._grim_db_label_tag = ""
        if self._button_checked(self.btn_phase):
            prior_phase_datasets = (
                list(getattr(figure, "_grim_held_phase_datasets", []))
                if hold and has_content
                else []
            )
            combined_phase_datasets = prior_phase_datasets + list(datasets)
            figure._grim_held_phase_datasets = combined_phase_datasets
            for note in plot_common.coherent_metadata_plot_warnings(
                combined_phase_datasets
            ):
                self._note_plot_render("Phase provenance note: " + note + ".")
        else:
            figure._grim_held_phase_datasets = []
        return True

    def _axis_selection_for_dataset(self, reference, dataset, axis: str, values):
        converted, tolerance = plot_common.selection_for_dataset(
            reference, dataset, axis, values
        )
        axis_values = {
            "azimuth": dataset.azimuths,
            "elevation": dataset.elevations,
            "frequency": dataset.frequencies,
        }[axis]
        try:
            return plot_common.unique_axis_selection(axis_values, converted, tolerance)
        except ValueError as exc:
            self._plot_selection_failed = True
            self._note_plot_render(
                f"Incompatible {axis} selection: {exc}. "
                "Choose one-to-one coordinates or explicitly regrid the dataset."
            )
            return None

    def _plot_axis_values(self, reference, dataset, axis: str, values):
        return plot_common.values_for_display(reference, dataset, axis, values)

    def _plot_axis_label(self, reference, axis: str) -> str:
        return plot_common.axis_label(reference, axis)

    def _plot_axis_name(self, reference, axis: str) -> str:
        if axis == "frequency":
            return "Frequency"
        return plot_common.angular_axis_name(reference, axis)

    def _plot_axis_unit(self, reference, axis: str) -> str:
        return plot_common.axis_unit(reference, axis)

    def _phase_p50(self, phase_degrees, axis=0):
        return plot_common.circular_median_degrees(phase_degrees, axis=axis)

    def _new_pbp_envelope(self):
        phase = self._button_checked(self.btn_phase)
        percentiles = self._pbp_percentiles()
        if phase:
            self._note_plot_render("Phase bands use a continuous branch of shortest containing arcs and may extend beyond ±180°.")
            if percentiles is not None:
                self._note_plot_render("Percentile PBP bands apply to magnitude; phase uses min–max arcs.")
                percentiles = None
        if percentiles is not None:
            self._note_plot_render("PBP percentiles interpolate in displayed units; missing samples are excluded, zero power is retained.")
        return plot_common.StreamingEnvelope(phase_degrees=phase, percentiles=percentiles,
                                             include_negative_infinity=not phase and not self._plot_scale_is_linear())

    def _configure_line_budget(self, candidate_count):
        """Allocate a display target across this render's visible curves."""
        if self._button_checked(self.btn_phase):
            return  # Do not add more scalar reduction to circular phase curves.
        visible = min(plot_common.MAX_LINE_SERIES, max(1, int(candidate_count)))
        if self._button_checked(self.btn_hold):
            visible = plot_common.MAX_LINE_SERIES
        self._plot_line_point_limit = max(4, min(
            plot_common.MAX_LINE_POINTS, plot_common.TOTAL_LINE_POINT_TARGET // visible
        ))

    def _plot_bounded_line(self, ax, x_values, y_values, *args, dataset=None,
                           trace_key=None, native_values=None, break_before=(), **kwargs):
        source_values = (x_values, y_values)
        x_values, y_values = plot_common.insert_plot_breaks(break_before, x_values, y_values)
        limit = getattr(self, "_plot_line_point_limit", plot_common.MAX_LINE_POINTS)
        x_display, y_display, decimated = plot_common.decimate_line(
            x_values, y_values, max_points=limit
        )
        if decimated:
            self._note_plot_render(
                f"Display curves use at most {limit:,} points each, preserving bucket extrema. "
                "Statistics and numerical exports use full data; figure exports use this display."
            )
        if trace_key is not None:
            trace_key = (self._dataset_plot_key(dataset),
                         getattr(self.plot_figure, "_grim_line_plot_signature", None), trace_key)
        label = kwargs.get("label")
        if (
            self._button_checked(getattr(self, "btn_hold", None))
            and isinstance(label, str)
            and label
            and not label.startswith("_")
        ):
            # Automatic refreshes and repeated clicks should replace the same
            # semantic series, not pile identical artists onto a held canvas.
            removed = False
            for existing in list(ax.lines):
                if trace_key is not None and getattr(existing, "_grim_trace_key", None) == trace_key:
                    self._release_artist_native_samples(existing)
                    existing.remove()
                    removed = True
            if removed:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            held_series = sum(
                1
                for existing in ax.lines
                if isinstance(existing.get_label(), str)
                and not existing.get_label().startswith("_")
            )
            if held_series >= plot_common.MAX_LINE_SERIES:
                self._note_plot_render(
                    f"Hold is capped at {plot_common.MAX_LINE_SERIES} visible series; "
                    "Clear the plot before adding different cuts."
                )
                return []
        if dataset is not None:
            kwargs.setdefault("linestyle", "-")
        if "marker" not in kwargs:
            for name, value in plot_common.singleton_line_markers(x_display, y_display).items():
                kwargs.setdefault(name, value)
        lines = ax.plot(x_display, y_display, *args, **kwargs)
        native_x, native_y = native_values if native_values is not None else source_values
        reduced = decimated or len(native_x) != len(x_display)
        for line in lines:
            line._grim_trace_key = trace_key
            line._grim_display_reduced = reduced
            if reduced:
                try:
                    line._grim_native_samples = plot_common.NativeSamples.curve(native_x, native_y)
                except (OSError, MemoryError, ValueError):
                    line._grim_native_probe_unavailable = True
                    self._note_plot_render("Native snapshot unavailable; markers use displayed samples only.")
        if dataset is not None:
            for line in lines:
                self._register_dataset_line(line, dataset)
                if self._native_db_overlay_enabled():
                    self._tag_native_db_artist(line, {self._native_db_quantity(dataset)})
        return lines

    def _bounded_plot_envelope(self, x_values, lower, upper, count=None):
        result = plot_common.decimate_envelope(x_values, lower, upper, count)
        x_display, lower_display, upper_display, count_display, decimated = result
        if decimated:
            self._note_plot_render(
                f"Bands over {plot_common.MAX_LINE_POINTS:,} samples were display-decimated; "
                "numerical exports use full data; figure exports use this display."
            )
        return x_display, lower_display, upper_display, count_display

    def _bounded_plot_image(self, x_values, y_values, image):
        x_display, y_display, image_display, decimated = plot_common.decimate_image(
            x_values, y_values, image
        )
        if decimated:
            self._note_plot_render(
                "Image shows block maxima; native cell probes use original samples. "
                "Numerical exports use full data; figure exports use this display."
            )
        return x_display, y_display, image_display

    def _native_image_snapshot(self, x_values, y_values, image, *, reduced):
        """Optional exact cursor data for a reduced ordinary image."""
        if not reduced:
            return None
        try:
            native = plot_common.NativeSamples.image(image)
            return (plot_common.coordinate_edges(x_values), plot_common.coordinate_edges(y_values), native)
        except (OSError, MemoryError, ValueError):
            self._note_plot_render("Native image snapshot unavailable; hover shows displayed block maxima only.")
            return False

    @staticmethod
    def _attach_native_image_snapshot(artist, snapshot):
        if snapshot is None:
            return
        artist._grim_display_reduced = True
        if snapshot is False:
            artist._grim_native_probe_unavailable = True
            return
        x_edges, y_edges, native = snapshot
        artist._grim_native_samples = native
        artist._grim_native_rectilinear_data = (x_edges, y_edges, native.values.T)

    def _zoom_target_axes(self) -> list:
        axes = self.plot_axes or [self.plot_ax]
        return [ax for ax in axes if ax is not None]

    def _sync_plot_limit_spins_from_axes(
        self,
        ax,
        *,
        sync_x: bool = True,
        sync_y: bool = True,
    ) -> None:
        if ax is None:
            return
        if (getattr(self.plot_figure, "_grim_independent_y", False)
                or getattr(self.plot_figure, "_grim_independent_x", False)):
            self.plot_figure._grim_limit_axes = ax

        if sync_x:
            if ax.name == "polar":
                xmin, xmax = plot_common.convert_axis_values(
                    ax.get_xlim(), "azimuth", "rad",
                    str(getattr(self, "_polar_display_unit", "deg")),
                )
            else:
                xmin, xmax = ax.get_xlim()
            if np.isfinite(xmin) and np.isfinite(xmax):
                self.spin_plot_xmin.blockSignals(True)
                self.spin_plot_xmax.blockSignals(True)
                plot_common.set_spin_value(self.spin_plot_xmin, float(xmin))
                plot_common.set_spin_value(self.spin_plot_xmax, float(xmax))
                self.spin_plot_xmin.blockSignals(False)
                self.spin_plot_xmax.blockSignals(False)

        if sync_y:
            ymin, ymax = ax.get_ylim()
            if np.isfinite(ymin) and np.isfinite(ymax):
                self.spin_plot_ymin.blockSignals(True)
                self.spin_plot_ymax.blockSignals(True)
                plot_common.set_spin_value(self.spin_plot_ymin, float(ymin))
                plot_common.set_spin_value(self.spin_plot_ymax, float(ymax))
                self.spin_plot_ymin.blockSignals(False)
                self.spin_plot_ymax.blockSignals(False)

        self._apply_plot_limits()

    def _clear_zoom_box_drag(self) -> None:
        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            self._zoom_box_drag = None
            return
        patch = drag.get("patch")
        canvas = drag.get("canvas")
        try:
            if patch is not None:
                patch.remove()
        except Exception:
            pass
        self._zoom_box_drag = None
        try:
            if canvas is not None:
                canvas.draw_idle()
        except Exception:
            pass

    def _clear_pan_drag(self, *, sync_limits: bool = False) -> None:
        drag = getattr(self, "_pan_drag", None)
        if not isinstance(drag, dict):
            self._pan_drag = None
            return
        ax = drag.get("ax")
        self._pan_drag = None
        if sync_limits and ax is not None:
            self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    @staticmethod
    def _uncheck_silently(btn) -> None:
        if btn is not None and btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _on_zoom_box_toggled(self, checked: bool) -> None:
        panel = getattr(self, "inline_analysis", None)
        if checked and panel is not None:
            panel.stop_region_selection()
        if checked and getattr(self, "spatial_overlays", None) is not None:
            self.spatial_overlays.stop_interaction()
        if not checked:
            self._clear_zoom_box_drag()
            return
        if self.plot_ax.name in ("polar", "3d"):
            self._uncheck_silently(getattr(self, "btn_zoom_box", None))
            self.status.showMessage("Box zoom is available on 2D rectilinear plots.")
            return
        # Zoom box, pan, and markers all claim the left button — one at a time.
        self._uncheck_silently(getattr(self, "btn_pan", None))
        self._uncheck_silently(getattr(self, "btn_markers", None))
        self.status.showMessage("Box zoom enabled. Drag left mouse on the plot to zoom.")

    def _on_pan_toggled(self, checked: bool) -> None:
        panel = getattr(self, "inline_analysis", None)
        if checked and panel is not None:
            panel.stop_region_selection()
        if checked and getattr(self, "spatial_overlays", None) is not None:
            self.spatial_overlays.stop_interaction()
        if not checked:
            self._clear_pan_drag()
            return
        if self.plot_ax.name in ("polar", "3d"):
            self._uncheck_silently(getattr(self, "btn_pan", None))
            self.status.showMessage("Pan is available on 2D rectilinear plots.")
            return
        self._uncheck_silently(getattr(self, "btn_zoom_box", None))
        self._uncheck_silently(getattr(self, "btn_markers", None))
        self._clear_zoom_box_drag()
        self.status.showMessage(
            "Pan enabled. Drag left mouse to move around the plot "
            "(middle-drag always pans)."
        )

    def _on_plot_scroll_zoom(self, event) -> None:
        if getattr(event, "canvas", None) is not self.plot_canvas:
            return
        ax = getattr(event, "inaxes", None)
        if ax not in self._zoom_target_axes():
            return
        if ax.name == "3d":
            return
        if getattr(self, "_zoom_box_drag", None):
            return
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and pan_drag.get("canvas") is event.canvas:
            return

        step = getattr(event, "step", None)
        if step is None:
            button = getattr(event, "button", None)
            if button == "up":
                step = 1.0
            elif button == "down":
                step = -1.0
            else:
                return
        if not np.isfinite(step) or np.isclose(step, 0.0):
            return

        zoom_base = 1.2
        zoom_factor = float(np.power(zoom_base, -step))
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0:
            return

        if ax.name == "polar":
            ymin, ymax = ax.get_ylim()
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                return
            ycenter = getattr(event, "ydata", None)
            if ycenter is None or not np.isfinite(ycenter):
                ycenter = 0.5 * (float(ymin) + float(ymax))
            new_ymin = float(ycenter) - (float(ycenter) - float(ymin)) * zoom_factor
            new_ymax = float(ycenter) + (float(ymax) - float(ycenter)) * zoom_factor
            if not np.isfinite(new_ymin) or not np.isfinite(new_ymax):
                return
            if np.isclose(new_ymin, new_ymax):
                return
            ax.set_ylim(new_ymin, new_ymax)
            self._sync_plot_limit_spins_from_axes(ax, sync_x=False, sync_y=True)
            return

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return
        xcenter = getattr(event, "xdata", None)
        ycenter = getattr(event, "ydata", None)
        if xcenter is None or not np.isfinite(xcenter):
            xcenter = 0.5 * (float(xmin) + float(xmax))
        if ycenter is None or not np.isfinite(ycenter):
            ycenter = 0.5 * (float(ymin) + float(ymax))

        new_xmin = float(xcenter) - (float(xcenter) - float(xmin)) * zoom_factor
        new_xmax = float(xcenter) + (float(xmax) - float(xcenter)) * zoom_factor
        new_ymin = float(ycenter) - (float(ycenter) - float(ymin)) * zoom_factor
        new_ymax = float(ycenter) + (float(ymax) - float(ycenter)) * zoom_factor
        if not all(np.isfinite(v) for v in (new_xmin, new_xmax, new_ymin, new_ymax)):
            return
        if np.isclose(new_xmin, new_xmax) or np.isclose(new_ymin, new_ymax):
            return

        ax.set_xlim(new_xmin, new_xmax)
        ax.set_ylim(new_ymin, new_ymax)
        self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    def _on_plot_mouse_press(self, event) -> None:
        if getattr(event, "canvas", None) is not self.plot_canvas:
            return
        button = getattr(event, "button", None)
        pan_requested = button in (MouseButton.MIDDLE, 2) or (
            button is MouseButton.LEFT
            and self._button_checked(getattr(self, "btn_pan", None))
        )
        if pan_requested:
            self._clear_zoom_box_drag()
            ax = getattr(event, "inaxes", None)
            if ax not in self._zoom_target_axes():
                return
            if ax.name in ("polar", "3d"):
                return
            x0 = getattr(event, "xdata", None)
            y0 = getattr(event, "ydata", None)
            if x0 is None or y0 is None or not np.isfinite(x0) or not np.isfinite(y0):
                return
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
                return
            self._pan_drag = {
                "ax": ax,
                "canvas": event.canvas,
                "x0": float(x0),
                "y0": float(y0),
                "xlim0": (float(xmin), float(xmax)),
                "ylim0": (float(ymin), float(ymax)),
            }
            return
        if button is not MouseButton.LEFT:
            return
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None and overlays.on_press(event):
            return
        if self._on_marker_press(event):
            return
        if not self._button_checked(getattr(self, "btn_zoom_box", None)):
            ax = getattr(event, "inaxes", None)
            delta = getattr(ax, "_grim_delta_map", None)
            if delta is not None:
                ax._grim_delta_pinned = delta_map_mode.cell_text(delta, event.xdata, event.ydata)
                self.hover_readout.setText(ax._grim_delta_pinned or "Outside Delta Map cells")
                return
            if getattr(self, "_active_plot_tab", "plotting") == "plotting":
                line = self._dataset_line_at_event(event)
                self._highlight_plot_dataset(
                    getattr(line, "_grim_dataset_key", None)
                )
            return
        ax = getattr(event, "inaxes", None)
        if ax not in self._zoom_target_axes():
            return
        if ax.name in ("polar", "3d"):
            self.status.showMessage("Box zoom is available on 2D rectilinear plots.")
            return
        x0 = getattr(event, "xdata", None)
        y0 = getattr(event, "ydata", None)
        if x0 is None or y0 is None or not np.isfinite(x0) or not np.isfinite(y0):
            return

        self._clear_zoom_box_drag()
        patch = Rectangle(
            (float(x0), float(y0)),
            0.0,
            0.0,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor=self._current_plot_text(),
            alpha=0.9,
        )
        ax.add_patch(patch)
        self._zoom_box_drag = {
            "ax": ax,
            "canvas": event.canvas,
            "x0": float(x0),
            "y0": float(y0),
            "patch": patch,
        }
        event.canvas.draw_idle()

    def _on_plot_mouse_move(self, event) -> None:
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None and overlays.on_motion(event):
            return
        if self._on_marker_motion(event):
            return
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and getattr(event, "canvas", None) is pan_drag.get("canvas"):
            ax = pan_drag.get("ax")
            if ax is None or getattr(event, "inaxes", None) is not ax:
                return
            x1 = getattr(event, "xdata", None)
            y1 = getattr(event, "ydata", None)
            if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
                return

            dx = float(x1) - float(pan_drag["x0"])
            dy = float(y1) - float(pan_drag["y0"])
            xmin0, xmax0 = pan_drag["xlim0"]
            ymin0, ymax0 = pan_drag["ylim0"]
            ax.set_xlim(float(xmin0) - dx, float(xmax0) - dx)
            ax.set_ylim(float(ymin0) - dy, float(ymax0) - dy)
            event.canvas.draw_idle()
            return

        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            return
        if getattr(event, "canvas", None) is not drag.get("canvas"):
            return

        ax = drag.get("ax")
        patch = drag.get("patch")
        if ax is None or patch is None:
            return
        if getattr(event, "inaxes", None) is not ax:
            return
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
            return

        x0 = float(drag["x0"])
        y0 = float(drag["y0"])
        x_min = min(x0, float(x1))
        x_max = max(x0, float(x1))
        y_min = min(y0, float(y1))
        y_max = max(y0, float(y1))
        patch.set_x(x_min)
        patch.set_y(y_min)
        patch.set_width(x_max - x_min)
        patch.set_height(y_max - y_min)
        event.canvas.draw_idle()

    def _on_plot_mouse_release(self, event) -> None:
        overlays = getattr(self, "spatial_overlays", None)
        if overlays is not None and overlays.on_release(event):
            return
        if self._on_marker_release(event):
            return
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and getattr(event, "canvas", None) is pan_drag.get("canvas"):
            sync = getattr(event, "button", None) in (MouseButton.MIDDLE, 2, MouseButton.LEFT)
            self._clear_pan_drag(sync_limits=sync)
            if getattr(event, "canvas", None) is not None:
                event.canvas.draw_idle()
            return

        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            return
        if getattr(event, "canvas", None) is not drag.get("canvas"):
            return

        ax = drag.get("ax")
        patch = drag.get("patch")
        if patch is not None:
            try:
                patch.remove()
            except Exception:
                pass
        self._zoom_box_drag = None

        if getattr(event, "button", None) is not MouseButton.LEFT:
            event.canvas.draw_idle()
            return
        if ax is None or getattr(event, "inaxes", None) is not ax:
            event.canvas.draw_idle()
            return
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
            event.canvas.draw_idle()
            return

        x0 = float(drag["x0"])
        y0 = float(drag["y0"])
        x_min = min(x0, float(x1))
        x_max = max(x0, float(x1))
        y_min = min(y0, float(y1))
        y_max = max(y0, float(y1))

        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        x_threshold = max(abs(float(cur_xmax) - float(cur_xmin)) * 0.005, 1e-12)
        y_threshold = max(abs(float(cur_ymax) - float(cur_ymin)) * 0.005, 1e-12)
        if (x_max - x_min) <= x_threshold or (y_max - y_min) <= y_threshold:
            event.canvas.draw_idle()
            return

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    def _apply_plot_limits(self) -> None:
        xmin = self.spin_plot_xmin.value()
        xmax = self.spin_plot_xmax.value()
        ymin = self.spin_plot_ymin.value()
        ymax = self.spin_plot_ymax.value()
        xstep = self.spin_plot_xstep.value()
        ystep = self.spin_plot_ystep.value()
        try:
            xmin, xmax = plot_common.finite_axis_limits(xmin, xmax)
            ymin, ymax = plot_common.finite_axis_limits(ymin, ymax)
        except ValueError as exc:
            self.status.showMessage(str(exc))
            return
        for spin, value in ((self.spin_plot_xmin, xmin), (self.spin_plot_xmax, xmax),
                            (self.spin_plot_ymin, ymin), (self.spin_plot_ymax, ymax)):
            plot_common.set_spin_value(spin, value)
        axes = self.plot_axes or [self.plot_ax]
        independent_x = getattr(self.plot_figure, "_grim_independent_x", False)
        independent_y = getattr(self.plot_figure, "_grim_independent_y", False)
        y_axis = getattr(self.plot_figure, "_grim_limit_axes", self.plot_ax)
        if y_axis not in axes:
            y_axis = self.plot_ax
        for ax in axes:
            ax.set_autoscale_on(False)
            if ax.name == "polar":
                # Always show the full polar grid; the x limits only choose
                # which azimuths get plotted, never a visible wedge.
                ax.set_thetamin(0.0)
                ax.set_thetamax(360.0)
                display_unit = str(getattr(self, "_polar_display_unit", "deg"))
                default_step = float(
                    plot_common.convert_axis_values(
                        [45.0], "azimuth", "deg", display_unit
                    )[0]
                )
                theta_step_display = xstep if xstep > 0.0 else default_step
                theta_step = float(
                    plot_common.convert_axis_values(
                        [theta_step_display], "azimuth", display_unit, "deg"
                    )[0]
                )
                ticks = plot_common.bounded_ticks(0.0, 360.0 - theta_step, theta_step)
                if ticks is None:
                    ticks = np.linspace(0.0, 360.0, plot_common.MAX_EXPLICIT_TICKS, endpoint=False)
                    self._note_plot_render(
                        f"Polar tick step requested over {plot_common.MAX_EXPLICIT_TICKS} ticks; "
                        "increase the X tick step."
                    )
                self._set_polar_thetagrids(ax, ticks)
            elif not independent_x or ax is y_axis:
                # Mixed diagnostic layouts have I/Q coordinates alongside a
                # frequency/angle sweep. Only the selected axis (and its
                # Matplotlib shared-X siblings) receives these bounds.
                ax.set_xlim(xmin, xmax)
                if xstep > 0.0:
                    ticks = plot_common.bounded_ticks(xmin, xmax, xstep)
                    if ticks is None:
                        ticks = np.linspace(xmin, xmax, plot_common.MAX_EXPLICIT_TICKS)
                        self._note_plot_render(
                            f"X tick step requested over {plot_common.MAX_EXPLICIT_TICKS} ticks; "
                            "increase the X tick step."
                        )
                    ax.set_xticks(ticks)
            if independent_y and ax is not y_axis:
                continue
            ax.set_ylim(ymin, ymax)
            if ystep > 0.0:
                ticks = plot_common.bounded_ticks(ymin, ymax, ystep)
                if ticks is None:
                    ticks = np.linspace(ymin, ymax, plot_common.MAX_EXPLICIT_TICKS)
                    self._note_plot_render(
                        f"Y tick step requested over {plot_common.MAX_EXPLICIT_TICKS} ticks; "
                        "increase the Y tick step."
                    )
                ax.set_yticks(ticks)
        self.plot_canvas.draw_idle()

    def _fit_both(self) -> None:
        if (getattr(self.plot_figure, "_grim_independent_x", False)
                or getattr(self.plot_ax, "_grim_data_limits", None) is not None):
            self._fit_independent_analysis_axes(scalex=True, scaley=True)
            return
        if self.plot_ax.name == "polar":
            self._fit_y()
            return
        self._fit_x()
        self._fit_y()

    def _effective_colormap(self) -> str:
        name = str(
            self.combo_colormap.currentData()
            or self.combo_colormap.currentText()
        )
        if self.chk_colormap_invert.isChecked():
            name = name + "_r"
        return name

    def _isar_window(self, n: int) -> np.ndarray:
        # Window math lives in isar_mode._window_array (Qt-free) so the ISAR
        # worker thread can use it; this wrapper just reads the combo.
        return isar_mode._window_array(self.combo_isar_window.currentText(), n)

    # ------------------------------------------------------------------
    # Async ISAR rendering: compute on a worker thread, display on the GUI
    # thread, coalesce bursts of requests (spinbox keystrokes, scrubbing).
    # ------------------------------------------------------------------

    def _invalidate_isar_figure(self) -> None:
        """Mark the visible ISAR canvas stale without discarding valid arrays."""

        self._isar_view_revision = int(
            getattr(self, "_isar_view_revision", 0)
        ) + 1
        self._last_isar_completed_view_revision = None
        self._last_isar_figure_token = None

    def _invalidate_isar_result(self) -> None:
        """Invalidate source/settings-bound ISAR state immediately.

        This is intentionally independent of the ordinary plotting render
        generation: the Plotting and ISAR tabs own different canvases.
        """

        self._cancel_plot_preparations()
        cancel_event = getattr(self, "_isar_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        self._isar_input_revision = int(
            getattr(self, "_isar_input_revision", 0)
        ) + 1
        self._invalidate_isar_figure()

    def _isar_figure_is_current(self) -> bool:
        return bool(
            getattr(self, "_last_isar_completed_view_revision", None)
            == getattr(self, "_isar_view_revision", 0)
            and getattr(self, "_last_isar_figure_token", None)
            is self.plot_figure
        )

    def _isar_submit(self, params: dict) -> None:
        # Do not let a prior canvas/result masquerade as the settings that are
        # now pending in the worker.
        queued = "isar_input_revision" in params
        if queued:
            if (
                params.get("isar_input_revision")
                != getattr(self, "_isar_input_revision", 0)
                or params.get("isar_view_revision")
                != getattr(self, "_isar_view_revision", 0)
            ):
                self.status.showMessage(
                    "Queued ISAR request discarded because its inputs or "
                    "settings changed before it started. Click Apply ISAR "
                    "Settings to form the current recipe."
                )
                return
        if not getattr(self, "_isar_busy", False):
            dataset_job_active = getattr(self, "_background_job_active", None)
            if callable(dataset_job_active) and dataset_job_active():
                self.status.showMessage(
                    "A dataset import, operation, save, or export is still running. "
                    "Wait for it to finish before starting ISAR reconstruction."
                )
                return
        if not queued:
            self._invalidate_isar_result()
            params["isar_input_revision"] = getattr(
                self, "_isar_input_revision", 0
            )
            params["isar_view_revision"] = getattr(
                self, "_isar_view_revision", 0
            )
        params.setdefault(
            "_record_python_request",
            bool(getattr(self, "_python_record_next_isar", False)),
        )
        cancel_event = threading.Event()
        params["cancel_check"] = cancel_event.is_set
        if getattr(self, "_isar_busy", False):
            # Latest request wins; it launches as soon as the current one lands.
            current_cancel = getattr(self, "_isar_cancel_event", None)
            if queued and current_cancel is not None:
                current_cancel.set()
            old_pending = getattr(self, "_isar_pending", None)
            if old_pending is not None:
                old_check = old_pending.get("_cancel_event")
                if old_check is not None:
                    old_check.set()
            params["_cancel_event"] = cancel_event
            self._isar_pending = params
            return
        self._isar_busy = True
        self._isar_cancel_event = cancel_event
        params["_cancel_event"] = cancel_event
        signals = getattr(self, "_isar_signals", None)
        if signals is None:
            signals = self._isar_signals = _IsarComputeSignals()
            signals.done.connect(self._on_isar_compute_done)
            signals.progress.connect(self._on_isar_progress)
        params["progress"] = lambda detail: signals.progress.emit(params, str(detail))
        self.status.showMessage("Computing ISAR image…")

        def work(params=params, signals=signals):
            try:
                result = isar_mode.compute_bands(params)
            except Exception as exc:  # surface, don't kill the worker silently
                result = f"ISAR computation failed: {exc}"
            signals.done.emit(params, result)

        QThreadPool.globalInstance().start(work)

    def _on_isar_compute_done(self, params: dict, result) -> None:
        self._isar_busy = False
        self._isar_cancel_event = None
        pending = getattr(self, "_isar_pending", None)
        self._isar_pending = None
        ok = not isinstance(result, str)
        superseded = pending is not None and params.get("_cancel_event") is not None \
            and params["_cancel_event"].is_set()
        if superseded:
            ok = False
        elif not ok:
            self.status.showMessage(result)
        elif (
            params.get("dataset") is not self.active_dataset
            or params.get("figure_token") is not self.plot_figure
            or params.get("isar_input_revision")
            != getattr(self, "_isar_input_revision", 0)
            or params.get("isar_view_revision")
            != getattr(self, "_isar_view_revision", 0)
        ):
            self.status.showMessage("ISAR result discarded (view changed while computing).")
            ok = False
        else:
            band_results, elapsed = result
            isar_mode.display_results(self, params, band_results, elapsed)
            overlays = getattr(self, "spatial_overlays", None)
            if overlays is not None:
                overlays.refresh()
            self._last_isar_figure_generation = params.get("render_generation")
            self._last_isar_completed_view_revision = params.get(
                "isar_view_revision"
            )
            self._last_isar_figure_token = params.get("figure_token")
            # Freeze the exact worker-captured recipe after every successful
            # async render, but emit it only for an explicit user request;
            # automatic refreshes stay out of the script.
            self._record_python_plot(
                "isar_image",
                resolved=self._resolved_isar_python_plot(params),
                datasets_override=[("ISAR Dataset", params["dataset"])],
                emit=bool(params.get("_record_python_request", False)),
            )
        drain_import = getattr(self, "_start_next_pending_import_batch", None)
        if callable(drain_import) and drain_import():
            if pending is not None:
                # Dataset imports take priority once the current reconstruction
                # reaches a safe boundary. Resume the latest coalesced ISAR
                # request after the queued import batches have drained.
                self._isar_pending = pending
            else:
                play_btn = getattr(self, "btn_isar_ap_play", None)
                if play_btn is not None and play_btn.isChecked():
                    play_btn.setChecked(False)
            return
        if pending is not None:
            self._isar_submit(pending)
            return
        # Cine mode: keep stepping while Play stays toggled and renders succeed.
        play_btn = getattr(self, "btn_isar_ap_play", None)
        if ok and play_btn is not None and play_btn.isChecked():
            generation = getattr(self, "_isar_play_generation", 0)
            QTimer.singleShot(30, lambda: self._isar_play_step(generation))

    # ------------------------------------------------------------------
    # Aperture scrubbing (center/width + step/play) and peak-relative clim
    # ------------------------------------------------------------------

    def _isar_step_aperture(self, direction: int) -> None:
        center_spin = getattr(self, "spin_isar_ap_center", None)
        width_spin = getattr(self, "spin_isar_ap_width", None)
        if center_spin is None or width_spin is None:
            return
        mode = "az_vs_range" if self.last_plot_mode == "az_vs_range" else "isar_image"
        chk = getattr(self, "chk_isar_aperture", None)
        width = float(width_spin.value())
        step = width / 2.0 if width > 0.0 else 1.0  # 50% overlap between looks
        new_center = float(np.mod(center_spin.value() + direction * step, 360.0))
        # Change both controls atomically, then form exactly one image in the
        # current mode. Their live signals would otherwise draw an intermediate
        # aperture and used to switch a range map into full ISAR.
        controls = [center_spin] + ([chk] if chk is not None else [])
        blocked = [widget.blockSignals(True) for widget in controls]
        try:
            if chk is not None:
                chk.setChecked(True)
            center_spin.setValue(new_center)
        finally:
            for widget, previous in zip(controls, blocked):
                widget.blockSignals(previous)
        self._invalidate_isar_result()
        if mode == "az_vs_range":
            self._plot_az_vs_range()
        else:
            self._plot_isar_image()

    def _on_isar_ap_prev(self) -> None:
        self._isar_step_aperture(-1)

    def _on_isar_ap_next(self) -> None:
        self._isar_step_aperture(+1)

    def _isar_play_step(self, generation=None) -> None:
        if generation is not None and generation != getattr(self, "_isar_play_generation", 0):
            return
        play_btn = getattr(self, "btn_isar_ap_play", None)
        if play_btn is None or not play_btn.isChecked():
            return
        mode = getattr(self, "_isar_play_mode", self.last_plot_mode)
        if self.last_plot_mode != mode:
            play_btn.setChecked(False)
            return
        previous_range_generation = getattr(self, "_last_az_vs_range_generation", None)
        self._isar_step_aperture(+1)
        # Range maps finish synchronously; coherent ISAR schedules its next
        # frame in the worker completion handler above.
        if mode == "az_vs_range" and play_btn.isChecked():
            generation = getattr(self, "_last_az_vs_range_generation", None)
            if generation is not None and generation != previous_range_generation:
                play_generation = getattr(self, "_isar_play_generation", 0)
                QTimer.singleShot(30, lambda: self._isar_play_step(play_generation))
            else:
                play_btn.setChecked(False)

    def _on_isar_ap_play(self, checked: bool) -> None:
        self._isar_play_generation = getattr(self, "_isar_play_generation", 0) + 1
        if checked:
            self._isar_play_mode = (
                "az_vs_range" if self.last_plot_mode == "az_vs_range" else "isar_image"
            )
            if self.last_plot_mode not in ("isar_image", "az_vs_range"):
                self.last_plot_mode = "isar_image"
            self._isar_play_step()

    def _on_isar_peak_scale(self) -> None:
        """Set the color scale to [peak − N, peak] of the rendered image —
        the standard ISAR dynamic-range convention. Works on the displayed
        data, so it's instant (no recompute)."""
        if self.last_plot_mode not in ("isar_image", "az_vs_range"):
            self.status.showMessage("Peak scaling applies to a rendered ISAR or azimuth/range image.")
            return
        meshes = self._current_isar_image_meshes()
        if not meshes:
            self.status.showMessage("No image to scale — plot one first.")
            return
        peak = float("-inf")
        for mesh in meshes:
            arr = np.asarray(mesh.get_array(), dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                peak = max(peak, float(finite.max()))
        if not np.isfinite(peak):
            return
        drop = float(self.spin_isar_peak_drop.value())
        if self._plot_scale_is_linear():
            # Linear display is image intensity |I|^2, so an N dB intensity
            # drop is a power ratio of 10^(-N/10).
            zmin = peak * 10.0 ** (-drop / 10.0)
        else:
            zmin = peak - drop
        self.spin_plot_zmin.blockSignals(True)
        self.spin_plot_zmax.blockSignals(True)
        self.spin_plot_zmin.setValue(zmin)
        self.spin_plot_zmax.setValue(peak)
        self.spin_plot_zmin.blockSignals(False)
        self.spin_plot_zmax.blockSignals(False)
        self._on_isar_clim_changed()
        self.status.showMessage(f"Image color scale set to peak − {drop:g} dB.")

    def _on_phase_toggled(self) -> None:
        if self.last_plot_mode == "delta_map" and self._button_checked(self.btn_phase):
            self.btn_phase.blockSignals(True)
            self.btn_phase.setChecked(False)
            self.btn_phase.blockSignals(False)
            self.status.showMessage("Delta Map shows level differences in dB. Use RF Compare for phase comparisons.")
            return
        self._on_polarization_selection_changed()
        self._maybe_autoplot()

    def _on_isar_window_changed(self) -> None:
        if self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()

    def _on_isar_clim_changed(self) -> None:
        """Retune the rendered ISAR image's color scale in place. Unlike the
        Apply button this does NOT re-run the imaging — clim changes only
        touch the existing AxesImage, so they can respond per keystroke."""
        if self.last_plot_mode not in ("isar_image", "az_vs_range"):
            return
        zmin = self.spin_plot_zmin.value()
        zmax = self.spin_plot_zmax.value()
        if zmin >= zmax:
            return
        meshes = self._current_isar_image_meshes()
        if not meshes:
            return
        for mesh in meshes:
            mesh.set_clim(zmin, zmax)
        for colorbar in self.plot_colorbars or []:
            self._apply_colorbar_ticks(colorbar)
        self._update_current_python_plot_style()
        self.plot_canvas.draw_idle()

    def _current_isar_image_meshes(self) -> list:
        if self.last_plot_mode == "az_vs_range":
            mesh = getattr(self, "_az_vs_range_mesh", None)
            candidates = [mesh] if mesh is not None else []
        else:
            candidates = getattr(self, "_isar_meshes", None) or []
        axes = self.plot_figure.axes
        return [mesh for mesh in candidates if mesh.axes in axes]


    def _fit_polar_y_range(self) -> tuple[float, float]:
        radial_values: list[np.ndarray] = []
        for line in self.plot_ax.lines:
            try:
                y = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if y.size == 0:
                continue
            finite = y[np.isfinite(y)]
            if finite.size:
                radial_values.append(finite)

        if radial_values:
            radial = np.concatenate(radial_values)
            ymin = float(np.nanmin(radial))
            ymax = float(np.nanmax(radial))
        else:
            ymin, ymax = self.plot_ax.get_ylim()
            ymin = float(ymin)
            ymax = float(ymax)

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return -1.0, 1.0
        if np.isclose(ymin, ymax):
            pad = max(1.0, abs(ymin) * 0.05)
            ymin -= pad
            ymax += pad
        return ymin, ymax

    def _waterfall_coordinate_limits(self, coordinate):
        """Fit image cells, excluding cut cursors from data bounds.

        Matplotlib relim() omits QuadMesh collections, so a lone crosshair
        otherwise collapses the corresponding image axis to one sample.
        """
        if self.last_plot_mode != "waterfall":
            return None
        edges = [artist._grim_rectilinear_data[coordinate]
                 for ax in (self.plot_axes or [self.plot_ax]) for artist in ax.collections
                 if getattr(artist, "_grim_image_record", None) is not None
                 and getattr(artist, "_grim_rectilinear_data", None) is not None]
        if not edges:
            return None
        return min(float(np.min(v)) for v in edges), max(float(np.max(v)) for v in edges)

    def _fit_independent_analysis_axes(self, *, scalex, scaley):
        """Fit mixed-coordinate diagnostics without fitting cursor overlays.

        Analysis renderers can retain native extrema as ``_grim_data_limits``
        (xmin, xmax, ymin, ymax), so display reduction and annotation artists
        cannot hide a narrow feature or expand the data bounds. Collections
        are included when no explicit bounds are available; relim omits them.
        """
        axes = self._zoom_target_axes()
        for ax in axes:
            bounds = getattr(ax, "_grim_data_limits", None)
            if bounds is not None and np.shape(bounds) == (4,) and np.all(np.isfinite(bounds)):
                xmin, xmax, ymin, ymax = bounds
                ax.dataLim.set_points(np.array([[xmin, ymin], [xmax, ymax]], dtype=float))
            else:
                ax.relim(visible_only=True)
                for collection in ax.collections:
                    if collection.get_visible():
                        points = collection.get_datalim(ax.transData).get_points()
                        if np.all(np.isfinite(points)):
                            ax.update_datalim(points)
            if scalex:
                ax.set_autoscalex_on(True)
            if scaley:
                ax.set_autoscaley_on(True)
        # Refresh every dataLim before shared axes compute their union.
        for ax in axes:
            ax.autoscale_view(scalex=scalex, scaley=scaley)
            if scalex:
                ax.set_autoscalex_on(False)
            if scaley:
                ax.set_autoscaley_on(False)
        for coordinate, enabled in (("x", scalex), ("y", scaley)):
            if enabled:
                spin = getattr(self, f"spin_plot_{coordinate}step")
                blocked = spin.blockSignals(True)
                spin.setValue(0.)
                spin.blockSignals(blocked)
        selected = getattr(self.plot_figure, "_grim_limit_axes", self.plot_ax)
        if selected not in axes:
            selected = self.plot_ax
        # Applying the toolbar writes both coordinates, so read both current
        # bounds even when only one dimension was fitted.
        self._sync_plot_limit_spins_from_axes(selected, sync_x=True, sync_y=True)

    def _fit_x(self) -> None:
        if (getattr(self.plot_figure, "_grim_independent_x", False)
                or getattr(self.plot_ax, "_grim_data_limits", None) is not None):
            self._fit_independent_analysis_axes(scalex=True, scaley=False)
            return
        if self.plot_ax.name == "polar":
            return

        delta = getattr(self.plot_ax, "_grim_delta_map", None)
        waterfall_limits = self._waterfall_coordinate_limits(0)
        if waterfall_limits is not None:
            self.plot_ax.set_xlim(*waterfall_limits)
        elif delta is not None:
            edges = delta.x_support.edges
            self.plot_ax.set_xlim(edges[0], edges[-1])
        else:
            self.plot_ax.set_autoscale_on(True)
            self.plot_ax.relim()
            self.plot_ax.autoscale_view(scalex=True, scaley=False)
        xmin, xmax = self.plot_ax.get_xlim()
        self.spin_plot_xmin.blockSignals(True)
        self.spin_plot_xmax.blockSignals(True)
        plot_common.set_spin_value(self.spin_plot_xmin, float(xmin))
        plot_common.set_spin_value(self.spin_plot_xmax, float(xmax))
        if self.spin_plot_xstep.value() > 0.0:
            self.spin_plot_xstep.blockSignals(True)
            self.spin_plot_xstep.setValue(0.0)
            self.spin_plot_xstep.blockSignals(False)
        self.spin_plot_xmin.blockSignals(False)
        self.spin_plot_xmax.blockSignals(False)
        self._apply_plot_limits()

    def _fit_y(self) -> None:
        if (getattr(self.plot_figure, "_grim_independent_x", False)
                or getattr(self.plot_ax, "_grim_data_limits", None) is not None):
            self._fit_independent_analysis_axes(scalex=False, scaley=True)
            return
        if getattr(self.plot_figure, "_grim_independent_y", False):
            for ax in self.plot_axes or [self.plot_ax]:
                ax.set_autoscaley_on(True)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
                ax.set_autoscaley_on(False)
            self.spin_plot_ystep.blockSignals(True)
            self.spin_plot_ystep.setValue(0.0)
            self.spin_plot_ystep.blockSignals(False)
            self._sync_plot_limit_spins_from_axes(self.plot_ax, sync_x=False, sync_y=True)
            return
        if self.plot_ax.name == "polar":
            ymin, ymax = self._fit_polar_y_range()
            self.spin_plot_ymin.blockSignals(True)
            self.spin_plot_ymax.blockSignals(True)
            plot_common.set_spin_value(self.spin_plot_ymin, float(ymin))
            plot_common.set_spin_value(self.spin_plot_ymax, float(ymax))
            if self.spin_plot_ystep.value() > 0.0:
                self.spin_plot_ystep.blockSignals(True)
                self.spin_plot_ystep.setValue(0.0)
                self.spin_plot_ystep.blockSignals(False)
            self.spin_plot_ymin.blockSignals(False)
            self.spin_plot_ymax.blockSignals(False)
            axes = self.plot_axes or [self.plot_ax]
            for ax in axes:
                ax.set_autoscale_on(False)
                ax.set_ylim(ymin, ymax)
            self.plot_canvas.draw_idle()
            return
        elif self._waterfall_coordinate_limits(1) is not None:
            ymin, ymax = self._waterfall_coordinate_limits(1)
            self.plot_ax.set_ylim(ymin, ymax)
        elif hasattr(self.plot_ax, "_grim_delta_map"):
            edges = self.plot_ax._grim_delta_map.y_support.edges
            ymin, ymax = edges[0], edges[-1]
            self.plot_ax.set_ylim(ymin, ymax)
        elif self.last_plot_mode == "cdf":
            # Probabilities always span the full 0-100 % scale.
            ymin, ymax = 0.0, 100.0
        else:
            self.plot_ax.set_autoscale_on(True)
            self.plot_ax.relim()
            self.plot_ax.autoscale_view(scalex=False, scaley=True)
            ymin, ymax = self.plot_ax.get_ylim()
        self.spin_plot_ymin.blockSignals(True)
        self.spin_plot_ymax.blockSignals(True)
        plot_common.set_spin_value(self.spin_plot_ymin, float(ymin))
        plot_common.set_spin_value(self.spin_plot_ymax, float(ymax))
        if self.spin_plot_ystep.value() > 0.0:
            self.spin_plot_ystep.blockSignals(True)
            self.spin_plot_ystep.setValue(0.0)
            self.spin_plot_ystep.blockSignals(False)
        self.spin_plot_ymin.blockSignals(False)
        self.spin_plot_ymax.blockSignals(False)
        self._apply_plot_limits()


    def _legend_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "loc": "upper right",
            "frameon": True,
            "framealpha": 0.92,
        }
        if self.last_plot_mode == "compare":
            kwargs["fontsize"] = 8
        return kwargs

    def _configure_legend(self, legend, ax=None) -> None:
        if legend is None:
            return
        if ax is None:
            ax = self.plot_ax
        # Reset a previously dragged/off-canvas legend. Legend.set_loc was
        # added after older Matplotlib releases still found on clusters, and
        # set_bbox_to_anchor's transform keyword also differs by release, so
        # keep the two operations independent and provide the stable numeric
        # upper-right fallback (loc code 1).
        try:
            legend.set_loc("upper right")
        except Exception:
            try:
                legend._loc = 1
            except Exception:
                pass
        try:
            legend.set_bbox_to_anchor(None)
        except Exception:
            try:
                legend._bbox_to_anchor = None
            except Exception:
                pass
        legend.set_visible(True)
        legend.set_zorder(1000)
        try:
            legend.set_in_layout(True)
        except AttributeError:
            pass
        for text in legend.get_texts():
            text.set_color(self._current_plot_text())
        frame = legend.get_frame()
        frame.set_visible(True)
        frame.set_alpha(0.92)
        frame.set_facecolor(self._current_plot_bg())
        frame.set_edgecolor(self._current_plot_grid())
        try:
            legend.set_draggable(True, use_blit=True, update="loc")
        except TypeError:
            try:
                legend.set_draggable(True, use_blit=True)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _format_hover_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if not np.isfinite(number):
            return "--"
        magnitude = abs(number)
        if magnitude >= 1e4 or (0.0 < magnitude < 1e-2):
            return f"{number:.2e}"
        return f"{number:.2f}"

    @staticmethod
    def _cursor_data_to_scalar(data) -> float | None:
        if data is None:
            return None
        try:
            if np.ma.isMaskedArray(data):
                data = data.compressed()
            values = np.asarray(data)
            if values.size == 0:
                return None
            if np.iscomplexobj(values):
                values = np.real(values)
            flat = np.asarray(values, dtype=float).ravel()
        except Exception:
            return None
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return None
        return float(finite[0])

    def _hover_z_from_axes(self, ax, event) -> float | None:
        self._hover_native_location = None
        self._hover_display_only = False
        artists = []
        artists.extend(reversed(getattr(ax, "collections", [])))
        artists.extend(reversed(getattr(ax, "images", [])))
        for artist in artists:
            probe = getattr(artist, "_grim_native_probe", None)
            if probe is not None:
                value = probe(getattr(event, "xdata", None), getattr(event, "ydata", None))
                if isinstance(value, tuple) and len(value) == 3:
                    self._hover_native_location = value[:2]
                    return self._cursor_data_to_scalar(value[2])
                if value is not None:
                    return self._cursor_data_to_scalar(value)
                continue
            rectilinear = getattr(artist, "_grim_native_rectilinear_data", None)
            if rectilinear is None:
                rectilinear = getattr(artist, "_grim_rectilinear_data", None)
            self._hover_display_only = bool(getattr(artist, "_grim_native_probe_unavailable", False))
            if rectilinear is not None:
                x_edges, y_edges, values = rectilinear
                x, y = getattr(event, "xdata", None), getattr(event, "ydata", None)
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                if not (x_edges[0] <= x <= x_edges[-1] and y_edges[0] <= y <= y_edges[-1]):
                    continue
                ix = min(len(x_edges) - 2, int(np.searchsorted(x_edges, x, side="right")) - 1)
                iy = min(len(y_edges) - 2, int(np.searchsorted(y_edges, y, side="right")) - 1)
                value = self._cursor_data_to_scalar(values[iy, ix])
                if value is not None:
                    return value
                continue  # A missing cell must not trigger polygon hit testing.
            getter = getattr(artist, "get_cursor_data", None)
            if getter is None:
                continue
            try:
                value = self._cursor_data_to_scalar(getter(event))
            except Exception:
                continue
            if value is not None:
                return value
        return None

    def _nearest_3d_hover_point(self, ax, event) -> tuple[float, float, float] | None:
        try:
            from mpl_toolkits.mplot3d import proj3d
        except Exception:
            return None
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None

        view_key = (
            round(float(getattr(ax, "elev", 0.0)), 3),
            round(float(getattr(ax, "azim", 0.0)), 3),
            tuple(np.round(np.asarray(ax.get_xlim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_ylim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_zlim3d(), dtype=float), 6)),
        )
        cache = getattr(ax, "_grim_hover_cache", None)
        if not isinstance(cache, dict) or cache.get("view_key") != view_key:
            xyz_chunks: list[np.ndarray] = []
            xy_chunks: list[np.ndarray] = []
            for artist in getattr(ax, "collections", []):
                offsets3d = getattr(artist, "_offsets3d", None)
                if offsets3d is None:
                    continue
                try:
                    xs = np.asarray(offsets3d[0], dtype=float).ravel()
                    ys = np.asarray(offsets3d[1], dtype=float).ravel()
                    zs = np.asarray(offsets3d[2], dtype=float).ravel()
                except Exception:
                    continue
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                if not np.any(finite):
                    continue
                xs = xs[finite]
                ys = ys[finite]
                zs = zs[finite]
                x2d, y2d, _ = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
                finite_2d = np.isfinite(x2d) & np.isfinite(y2d)
                if not np.any(finite_2d):
                    continue
                xs = xs[finite_2d]
                ys = ys[finite_2d]
                zs = zs[finite_2d]
                x2d = x2d[finite_2d]
                y2d = y2d[finite_2d]
                xy_pixels = ax.transData.transform(np.column_stack([x2d, y2d]))
                xyz_chunks.append(np.column_stack([xs, ys, zs]))
                xy_chunks.append(xy_pixels)
            if not xyz_chunks or not xy_chunks:
                return None
            cache = {
                "view_key": view_key,
                "xyz": np.vstack(xyz_chunks),
                "xy": np.vstack(xy_chunks),
            }
            setattr(ax, "_grim_hover_cache", cache)

        xy_pixels = cache.get("xy")
        xyz_points = cache.get("xyz")
        if xy_pixels is None or xyz_points is None or len(xy_pixels) == 0:
            return None

        distances = np.hypot(xy_pixels[:, 0] - event.x, xy_pixels[:, 1] - event.y)
        finite = np.isfinite(distances)
        if not np.any(finite):
            return None
        idx = int(np.argmin(np.where(finite, distances, np.inf)))
        if distances[idx] > 24.0:
            return None
        x_val, y_val, z_val = xyz_points[idx]
        return float(x_val), float(y_val), float(z_val)

    def _reset_hover_readout(self, hover_readout=None) -> None:
        # A leave event supersedes any pending hover update.
        timer = getattr(self, "_hover_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_hover = None
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        label.setText(getattr(getattr(self, "plot_ax", None), "_grim_delta_pinned", None) or "x: --   y: --")

    def _schedule_hover(self, event, hover_readout=None) -> None:
        self._pending_hover = (event, hover_readout)
        timer = getattr(self, "_hover_timer", None)
        if timer is None:
            self._flush_hover()
            return
        if not timer.isActive():
            timer.start()

    def _flush_hover(self) -> None:
        pending = self._pending_hover
        self._pending_hover = None
        if pending is None:
            return
        event, label = pending
        self._on_plot_hover(event, label)

    def _on_plot_hover(self, event, hover_readout=None) -> None:
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None:
            self._reset_hover_readout(label)
            return
        delta = getattr(ax, "_grim_delta_map", None)
        if delta is not None:
            label.setText(delta_map_mode.cell_text(delta, event.xdata, event.ydata) or "Outside Delta Map cells")
            return
        # z stays on the SAME line as x/y: a second line changes the label
        # height and visibly shifts the canvas whenever the cursor crosses
        # onto/off a z-valued artist.
        if ax.name == "3d":
            point = self._nearest_3d_hover_point(ax, event)
            if point is None:
                label.setText("x: --   y: --   z: --")
                return
            x_val, y_val, z_val = point
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
                f"   z: {self._format_hover_number(z_val)}"
            )
            return

        x_val = getattr(event, "xdata", None)
        y_val = getattr(event, "ydata", None)
        if (
            x_val is None
            or y_val is None
            or not np.isfinite(x_val)
            or not np.isfinite(y_val)
        ):
            self._reset_hover_readout(label)
            return

        z_val = self._hover_z_from_axes(ax, event)
        peak_location = getattr(self, "_hover_native_location", None)
        if peak_location is not None:
            label.setText(
                f"Overview block peak at x: {self._format_hover_number(peak_location[0])}"
                f"   y: {self._format_hover_number(peak_location[1])}"
                f"   z: {self._format_hover_number(z_val)}"
            )
            return
        if z_val is None:
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
            )
            return
        label.setText(
            f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
            f"   z: {self._format_hover_number(z_val)}"
            + (" (displayed block maximum; native probe unavailable)" if self._hover_display_only else "")
        )

    def _update_legend_visibility(self, checked: bool | None = None) -> None:
        """Create/show/hide legends on every current line-plot axis.

        ``checked`` is accepted directly from the toolbar signal. This avoids
        reading the other tab's toggle during a tab-context transition; plot
        renderers call the method without it and use the active toggle.
        """
        show = bool(self.chk_plot_legend.isChecked()) if checked is None else bool(checked)
        self._update_current_python_plot_style(show_legend=show)
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._refresh_native_db_labels(ax)
            legend = ax.get_legend()
            handles, labels = ax.get_legend_handles_labels()
            if not show or not handles:
                if legend is not None:
                    legend.set_visible(False)
                continue
            existing_labels = (
                [text.get_text() for text in legend.get_texts()]
                if legend is not None else None
            )
            if legend is None or existing_labels != labels:
                if legend is not None:
                    legend.remove()
                legend = ax.legend(handles, labels, **self._legend_kwargs())
            self._configure_legend(legend, ax)
            self._sync_dataset_legend(ax)
        # Rendering paths call _apply_plot_limits immediately afterward, which
        # already schedules one draw. A direct toolbar toggle still needs a
        # repaint, but draw_idle coalesces it with any pending canvas work.
        if checked is not None:
            self.plot_canvas.draw_idle()

    def _on_explicit_isar_plot_clicked(self, _checked: bool = False) -> None:
        """Form one ISAR request with recorder intent scoped to that request."""

        self._python_record_next_isar = True
        try:
            self._plot_isar_image()
        finally:
            self._python_record_next_isar = False

    def _on_explicit_plot_clicked(self, mode: str) -> None:
        """Emit a synchronous plot after its renderer reports success."""
        if getattr(self, "_ordinary_plot_pending", False):
            self._ordinary_plot_emit = True
            return
        if "updated" in str(self.status.currentMessage()).lower():
            self._emit_last_successful_python_plot()

    def _set_compare_sector_controls_visible(self, visible: bool) -> None:
        controls = getattr(self, "compare_sector_bar", None)
        if controls is not None:
            controls.setVisible(bool(visible))

    def _on_compare_sector_controls_changed(self, *_args) -> None:
        """Re-render Compare after a committed sector/display change."""

        if getattr(self, "last_plot_mode", None) == "compare":
            self._plot_compare()

    def _on_delta_map_controls_changed(self, *_args) -> None:
        if getattr(self, "last_plot_mode", None) == "delta_map":
            self._plot_delta_map()

    def _capture_successful_python_plot(self, mode: str) -> None:
        """Freeze the semantic spec without recording automatic re-plots."""

        if not getattr(self, "_ordinary_plot_pending", False) and "updated" in str(self.status.currentMessage()).lower():
            self._record_python_plot(mode, emit=False)

    def _update_current_python_plot_style(
        self, *, show_legend: bool | None = None
    ) -> None:
        """Overlay live view-only controls on this canvas's frozen recipe."""

        spec = getattr(self, "last_python_plot_spec", None)
        if not spec or spec[0] != "supported":
            return
        parameters = dict(spec[4])
        parameters["colormap"] = self._effective_colormap()
        parameters["show_grid"] = self._plot_grid_enabled()
        parameters["show_legend"] = (
            bool(self.chk_plot_legend.isChecked())
            if show_legend is None
            else bool(show_legend)
        )
        if spec[3] == "isar_image":
            parameters.update(self._current_isar_python_display_style())
        if spec[3] == "delta_map":
            parameters["show_colorbar"] = bool(self.chk_colorbar.isChecked())
        if spec[3] == "waterfall":
            parameters["show_colorbar"] = bool(self.chk_colorbar.isChecked())
            parameters["shared_colorbar"] = bool(self.chk_colorbar_shared.isChecked())
            zmin, zmax = float(self.spin_plot_zmin.value()), float(self.spin_plot_zmax.value())
            parameters["color_limits"] = (zmin, zmax) if zmin < zmax else None
            controls = getattr(self, "image_analysis_controls", None)
            if controls is not None:
                options = dict(parameters.get("heatmap_options", {}))
                options["cut_position"] = controls.options().get("cut_position")
                parameters["heatmap_options"] = options
        self.last_python_plot_spec = (*spec[:4], parameters)

    def _current_isar_python_display_style(self) -> dict[str, object]:
        """Capture GUI display controls supported by headless ISAR replay."""

        zmin = float(self.spin_plot_zmin.value())
        zmax = float(self.spin_plot_zmax.value())
        color_limits = (zmin, zmax) if zmin < zmax else None
        meshes = [
            mesh
            for mesh in getattr(self, "_isar_meshes", None) or []
            if getattr(mesh, "axes", None) is not None
        ]
        if meshes:
            mesh_limits = [tuple(float(value) for value in mesh.get_clim()) for mesh in meshes]
            if all(limits == mesh_limits[0] for limits in mesh_limits[1:]):
                color_limits = mesh_limits[0]
            else:
                # Per-band automatic normalization has no single explicit
                # clamp; replay it by leaving color_limits unset.
                color_limits = None
        return {
            "show_colorbar": bool(self.chk_colorbar.isChecked()),
            "shared_colorbar": bool(self.chk_colorbar_shared.isChecked()),
            "square_aspect": bool(self.chk_isar_square.isChecked()),
            "color_limits": color_limits,
            "color_step": float(self.spin_plot_zstep.value()),
        }

    def _record_python_plot(
        self,
        mode: str,
        *,
        resolved: dict[str, object] | None = None,
        datasets_override=None,
        emit: bool = True,
    ) -> None:
        recorder = getattr(self, "python_recorder", None)
        reference_getter = getattr(self, "_python_reference_for_dataset", None)
        if recorder is None or not callable(reference_getter):
            return

        from GRIM_Backend.plotting.registry import HEADLESS_PLOT_MODES
        supported_modes = HEADLESS_PLOT_MODES
        if mode not in supported_modes:
            spec = (
                "unsupported",
                mode,
                "the recorder supports rectangular/polar azimuth, frequency, "
                "elevation-sweep, CDF, waterfall, Delta Map, and ISAR plots only",
            )
            self.last_python_plot_spec = spec
            if emit:
                recorder.record_unsupported_plot(spec[1], spec[2])
            return
        if mode not in ("delta_map", "cdf", "waterfall") and self._button_checked(getattr(self, "btn_pbp", None)):
            spec = (
                "unsupported",
                mode,
                "PBP rendering does not yet have a matching headless implementation",
            )
            self.last_python_plot_spec = spec
            if emit:
                recorder.record_unsupported_plot(spec[1], spec[2])
            return
        if mode not in ("delta_map", "waterfall") and self._button_checked(getattr(self, "btn_hold", None)):
            spec = (
                "unsupported",
                mode,
                "Hold overlays depend on prior plot state and are intentionally "
                "outside this simple headless recorder",
            )
            self.last_python_plot_spec = spec
            if emit:
                recorder.record_unsupported_plot(spec[1], spec[2])
            return

        datasets = (
            list(datasets_override)
            if datasets_override is not None
            else self._selected_datasets()
        )
        references = []
        names = []
        for name, dataset in datasets:
            reference = reference_getter(dataset)
            if reference is None:
                return
            references.append(reference)
            names.append(name)
        if not references:
            return

        override = dict(resolved or {})
        azimuths = override.pop("azimuths", self._selected_values(self.list_az))
        elevations = override.pop("elevations", self._selected_values(self.list_elev))
        frequencies = override.pop(
            "frequencies", self._selected_values(self.list_freq)
        )
        polarization = override.pop("polarization", None)
        if polarization is None:
            selected_pol = self._selected_values(self.list_pol)
            if not selected_pol or (mode in ("isar_image", "delta_map") and len(selected_pol) != 1):
                return
            polarization = selected_pol[0] if len(selected_pol) == 1 else tuple(selected_pol)

        parameters: dict[str, object] = {
            "azimuths": list(azimuths),
            "elevations": list(elevations),
            "frequencies": list(frequencies),
            "polarization": polarization,
            # The parameter-list values live in the active/reference
            # dataset's units.  Table selection order is independent of the
            # active row, so replay must not assume datasets[0] is that frame.
            "reference_index": next(
                (
                    index
                    for index, (_name, dataset) in enumerate(datasets)
                    if dataset is self.active_dataset
                ),
                0,
            ),
            "phase": self._button_checked(getattr(self, "btn_phase", None)),
            "scale": self._plot_scale_mode(),
            "colormap": self._effective_colormap(),
            "show_grid": self._plot_grid_enabled(),
            "show_legend": bool(self.chk_plot_legend.isChecked()),
            "polar_zero": self._polar_zero_location(),
        }
        parameters.update(override)
        analysis = getattr(self, "analysis_controls", None)
        if mode in ("frequency", "elevation_sweep") and callable(getattr(analysis, "sweep_aggregation", None)):
            parameters["sweep_options"] = analysis.sweep_aggregation()
        if mode == "cdf" and callable(getattr(analysis, "cdf_options", None)):
            parameters["cdf_options"] = analysis.cdf_options()
        image_analysis = getattr(self, "image_analysis_controls", None)
        if mode == "waterfall" and callable(getattr(image_analysis, "options", None)):
            parameters["heatmap_options"] = image_analysis.options()
            parameters["show_colorbar"] = bool(self.chk_colorbar.isChecked())
            parameters["shared_colorbar"] = bool(self.chk_colorbar_shared.isChecked())
            parameters["color_limits"] = (
                (float(self.spin_plot_zmin.value()), float(self.spin_plot_zmax.value()))
                if self.spin_plot_zmin.value() < self.spin_plot_zmax.value() else None
            )
        if mode == "isar_image":
            parameters.update(self._current_isar_python_display_style())
        spec = (
            "supported",
            tuple(references),
            tuple(names),
            mode,
            parameters,
        )
        self.last_python_plot_spec = spec
        if emit:
            recorder.record_plot(
                spec[1],
                names=spec[2],
                mode=spec[3],
                parameters=spec[4],
            )

    @staticmethod
    def _resolved_isar_python_plot(params: dict) -> dict[str, object]:
        """Freeze the physical selectors and formation recipe that succeeded.

        ISAR completes asynchronously, so reading the live widgets here could
        record settings the user changed after the worker started.  Everything
        needed for replay is instead taken from the worker's captured params.
        """

        dataset = params["dataset"]
        azimuth_indices = sorted(
            {int(index) for band in params["bands"] for index in band}
        )
        frequency_indices = [int(index) for index in params["freq_indices_sorted"]]
        target = params.get("az_target_deg")
        options: dict[str, object] = {
            "window": str(params["window_name"]),
            "reconstruction": str(params["recon"]),
            "length_unit": str(params["unit_name"]),
            "aperture_center_degrees": params.get("az_center_deg"),
            "azimuth_target_degrees": (
                None if target is None else np.asarray(target, dtype=float).tolist()
            ),
            "l1_strength": float(params["l1_strength"]),
            "l1_iterations": int(params["l1_iters"]),
            "flip_x": bool(params["flip_x"]),
            "flip_y": bool(params["flip_y"]),
        }
        for key in ("aperture_mode", "scene_half_extent_m", "composite_side", "native_diagnostics", "memory_limit_bytes", "memory_allow_exceed"):
            if key in params:
                options[key] = params[key]
        return {
            "azimuths": np.asarray(dataset.azimuths)[azimuth_indices].tolist(),
            "elevations": [
                np.asarray(dataset.elevations)[int(params["elev_idx"])].item()
            ],
            "frequencies": np.asarray(dataset.frequencies)[frequency_indices].tolist(),
            "polarization": str(
                np.asarray(dataset.polarizations)[int(params["pol_idx"])]
            ),
            "phase": False,
            "isar_options": options,
        }

    def _drop_dataset_from_python_plot_spec(self, key) -> None:
        """Keep the frozen recipe equal to the canvas after a manual removal."""

        spec = getattr(self, "last_python_plot_spec", None)
        if not spec or spec[0] != "supported":
            return
        _status, references, names, mode, parameters = spec
        index = next(
            (i for i, ref in enumerate(references) if ref.dataset_id == key), None
        )
        if index is None:
            return
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None:
            recorder.invalidate_current_plot()
        reference_index = int(parameters.get("reference_index", 0))
        if index == reference_index:
            # Selector values are expressed in this dataset's units.
            self.last_python_plot_spec = (
                "unsupported",
                mode,
                "the dataset that defined the plot selection was removed from "
                "the canvas; plot again to record this view",
            )
            return
        keep = [i for i in range(len(references)) if i != index]
        parameters = dict(parameters)
        parameters["reference_index"] = reference_index - int(index < reference_index)
        self.last_python_plot_spec = (
            "supported",
            tuple(references[i] for i in keep),
            tuple(names[i] for i in keep),
            mode,
            parameters,
        )

    def _emit_last_successful_python_plot(self) -> bool:
        """Emit the frozen spec corresponding to the visible plot canvas."""

        recorder = getattr(self, "python_recorder", None)
        overlays = getattr(self, "spatial_overlays", None)
        if recorder is not None and overlays is not None and overlays.has_visible_paths():
            recorder.record_unsupported_plot(
                "spatial_overlay", "editable overlays are visual annotations; use Export Plot for the annotated image and Save XY for the points"
            )
            return False
        spec = getattr(self, "last_python_plot_spec", None)
        if recorder is None or not spec:
            if recorder is not None:
                recorder.invalidate_current_plot()
            return False
        if spec[0] == "unsupported":
            recorder.record_unsupported_plot(spec[1], spec[2])
            return False
        recorder.record_plot(
            spec[1],
            names=spec[2],
            mode=spec[3],
            parameters=spec[4],
        )
        return True

    def _plot_azimuth_rect(self) -> None:
        self._set_compare_sector_controls_visible(False)
        azimuth_rect_mode.render(self)
        self._capture_successful_python_plot("azimuth_rect")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_azimuth_polar(self) -> None:
        self._set_compare_sector_controls_visible(False)
        azimuth_polar_mode.render(self)
        self._capture_successful_python_plot("azimuth_polar")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_frequency(self) -> None:
        self._set_compare_sector_controls_visible(False)
        if frequency_mode.render(self):
            return
        self._capture_successful_python_plot("frequency")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_elevation_sweep(self) -> None:
        self._set_compare_sector_controls_visible(False)
        if elevation_sweep_mode.render(self):
            return
        self._capture_successful_python_plot("elevation_sweep")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_isar_image(self) -> None:
        self._set_compare_sector_controls_visible(False)
        isar_mode.render(self)
        self._maybe_autoscale()

    def _plot_az_vs_range(self) -> None:
        self._set_compare_sector_controls_visible(False)
        az_vs_range_mode.render(self)
        self._capture_successful_python_plot("az_vs_range")
        self._maybe_autoscale()

    def _plot_waterfall(self) -> None:
        self._set_compare_sector_controls_visible(False)
        if waterfall_mode.render(self):
            return
        self._capture_successful_python_plot("waterfall")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_compare(self) -> None:
        compare_mode.render(self)
        self._capture_successful_python_plot("compare")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_cdf(self) -> None:
        self._set_compare_sector_controls_visible(False)
        if cdf_mode.render(self):
            return
        self._capture_successful_python_plot("cdf")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_range_freq(self) -> None:
        self._set_compare_sector_controls_visible(False)
        if range_freq_mode.render(self):
            return
        self._capture_successful_python_plot("range_freq")
        self._maybe_autoscale()

    def _plot_sector_stats(self) -> None:
        self._set_compare_sector_controls_visible(False)
        sector_stats_mode.render(self)
        self._capture_successful_python_plot("sector_stats")
        self._restore_plot_markers()
        self._maybe_autoscale()

    def _plot_delta_map(self) -> None:
        if self._button_checked(self.btn_phase):
            self.btn_phase.blockSignals(True)
            self.btn_phase.setChecked(False)
            self.btn_phase.blockSignals(False)
            self._on_polarization_selection_changed()
        delta_map_mode.render(self)

    def _ensure_compare_axes(self):
        """Return (top_ax, res_ax) for the 2-panel compare layout, recreating if needed."""
        if (
            self.plot_axes is not None
            and len(self.plot_axes) == 1
            and len(self.plot_figure.axes) == 2
        ):
            return self.plot_figure.axes[0], self.plot_figure.axes[1]
        self._remove_colorbar()
        self.plot_figure.clear()
        top_ax, res_ax = self.plot_figure.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        self.plot_ax = top_ax
        # plot_axes = [top_ax] so _apply_plot_limits only touches the top axis
        # (x-limits propagate automatically via sharex; residual y auto-scales)
        self.plot_axes = [top_ax]
        self.plot_figure.set_facecolor(self._current_plot_bg())
        return top_ax, res_ax

    def _on_isar_progress(self, params, detail):
        if params.get("isar_input_revision") == getattr(self, "_isar_input_revision", 0):
            self.status.showMessage("ISAR: " + detail)

    def _isar_control_changed(self, context, widget):
        if not widget.isEnabled():
            return
        if widget in (context.spin_isar_freq_min, context.spin_isar_freq_max) and not context.chk_isar_freq_band.isChecked():
            return
        if widget in (context.spin_isar_az_min, context.spin_isar_az_max, context.spin_isar_az_step) and not context.chk_isar_az_interp.isChecked():
            return
        if widget in (context.spin_isar_ap_center, context.spin_isar_ap_width) and not context.chk_isar_aperture.isChecked():
            return
        self._invalidate_isar_result()
