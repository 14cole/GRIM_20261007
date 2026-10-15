"""Down-range profiles over sliding frequency sub-bands.

Each column is the range profile of one sub-band: the selected complex
sweeps are windowed, inverse-FFT'd with zero padding, and their power is
averaged over the selected azimuths and elevations. A point scatterer stays at
one range in every sub-band; cavities, traveling waves, and resonances drift
in range or appear only in part of the band. Profiles are scaled by the
window's coherent gain, so an isolated point shows its own level at its peak.
"""

from __future__ import annotations

import numpy as np

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.transforms import _declared_time_sign
from GRIM_Backend.datasets.result_storage import allocate_result, operation_result_storage, result_blocks
from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest, PlotResult, check_cancel

from . import common
from .azimuth_rect_mode import _plan_series
from .isar_mode import _window_array

MAX_COLUMNS = 200
MAX_BATCH_BYTES = 32 * 1024**2
RANGE_UNITS = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "in": 1.0 / 0.0254, "ft": 1.0 / 0.3048}


def subband_starts(count: int, width: int, max_columns: int = MAX_COLUMNS) -> list[int]:
    """Bounded sub-band starts including both ends of the selected sweep."""
    if not 1 <= width <= count or max_columns < 1:
        raise ValueError("Sub-band width and column limit must be positive and fit the selection.")
    positions = count - width + 1
    if positions > 1 and max_columns < 2:
        raise ValueError("At least two columns are needed to cover both ends of the selection.")
    return np.linspace(0, positions - 1, min(positions, max_columns), dtype=int).tolist()


def range_frequency_plan(n_sweeps, count, width, *, batch_bytes=MAX_BATCH_BYTES):
    """Plan memory from bytes per batch, independent of total transform work.

    A complete selected sweep and one padded FFT vector are unavoidable RAM
    work. Additional sweeps are batched within the requested scratch budget.
    """
    starts = subband_starts(count, width)
    size = max(64, 1 << (4 * int(width) - 1).bit_length())
    bytes_per_sweep = 32 * int(count) + 32 * int(width) + 48 * size
    batch_rows = max(1, min(max(1, int(n_sweeps)), int(batch_bytes) // bytes_per_sweep))
    return dict(starts=starts, fft_size=size, batch_rows=batch_rows,
                scratch_bytes=batch_rows * bytes_per_sweep,
                output_bytes=len(starts) * size * 8,
                transform_cells=int(n_sweeps) * len(starts) * size)


class SelectedSweeps:
    """Read only a bounded batch, retaining the azimuth/elevation sweep order."""
    def __init__(self, dataset, selection):
        self.dataset, self.selection = dataset, selection
        self.shape = (len(selection[0]) * len(selection[1]), len(selection[2]))
        self.complete_count = 0
        self.conjugate = _declared_time_sign(dataset) == "-jwt"

    def __getitem__(self, rows):
        azimuth, elevation, frequency, polarization = self.selection
        indices = np.arange(*rows.indices(self.shape[0]))
        values = self.dataset.rcs_slice((
            np.asarray(azimuth)[indices // len(elevation), None],
            np.asarray(elevation)[indices % len(elevation), None],
            np.asarray(frequency)[None, :], int(polarization[0])))
        if self.conjugate:
            np.conjugate(values, out=values)
        return values


def range_frequency_map(sweeps, frequencies_hz, *, width: int, window: str,
                        batch_bytes=MAX_BATCH_BYTES, cancel=None):
    """Return ``(ranges_m, starts, power)`` with power shaped (sub-bands, ranges).

    ``sweeps`` are complex (n_sweeps, n_frequencies) arrays on a uniform
    frequency grid in GRIM's exp(+jwt) law. Incomplete sweeps are omitted as
    entire sweeps, so every sub-band averages the same population.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    count = frequencies_hz.size
    if not hasattr(sweeps, "shape"):
        sweeps = np.asarray(sweeps)
    if len(sweeps.shape) != 2 or sweeps.shape[1] != count:
        raise ValueError("Range–frequency sweeps must match the frequency axis")
    problem = _uniform_problem(frequencies_hz)
    if problem:
        raise ValueError(f"Range–frequency {problem}")
    plan = range_frequency_plan(sweeps.shape[0], count, width, batch_bytes=batch_bytes)
    step_hz = float(np.mean(np.diff(frequencies_hz)))
    taper = _window_array(window, width)
    gain = float(np.sum(taper))
    if gain <= 0 or not np.isfinite(gain):
        raise ValueError("Range–frequency window has no coherent gain")
    size, starts = plan["fft_size"], plan["starts"]
    check_cancel(cancel)
    power = allocate_result((len(starts), size), np.float64, fill=0.)
    complete_count = 0
    for first in range(0, sweeps.shape[0], plan["batch_rows"]):
        check_cancel(cancel)
        batch = np.asarray(sweeps[first:first + plan["batch_rows"]], dtype=np.complex128)
        complete = np.all(np.isfinite(batch), axis=-1)
        if not np.all(complete):
            batch = batch[complete]
        complete_count += len(batch)
        if not len(batch):
            continue
        for column, start in enumerate(starts):
            check_cancel(cancel)
            profile = np.fft.ifft(batch[:, start:start + width] * taper, n=size, axis=-1)
            profile *= size / gain
            intensity = np.abs(profile)
            np.square(intensity, out=intensity)
            power[column] += np.sum(intensity, axis=0)
            del profile, intensity
        del batch
    if isinstance(sweeps, SelectedSweeps):
        sweeps.complete_count = complete_count
    if not complete_count:
        raise ValueError("no complete complex sweep; needs phase")
    for column in range(len(starts)):
        check_cancel(cancel)
        power[column] = np.fft.fftshift(power[column]) / complete_count
    ranges = np.fft.fftshift(np.fft.fftfreq(size, d=step_hz)) * (C0 / 2.0)
    return ranges, starts, power


def _uniform_problem(frequencies_hz) -> str:
    if frequencies_hz.size < 8:
        return "needs at least 8 selected frequencies"
    if not np.all(np.isfinite(frequencies_hz)):
        return "needs finite selected frequencies"
    steps = np.diff(frequencies_hz)
    step = float(np.mean(steps))
    if step <= 0.0 or np.max(np.abs(steps - step)) > 1.0e-6 * step:
        return "needs uniformly spaced selected frequencies"
    return ""


def prepare_range_frequency(request, cancel=None):
    """No Qt access: bounded reads/FFTs, conversion, reduction and probes."""
    result = PlotResult(request, [])
    percent, window, unit = (request.options[key] for key in ("percent", "window", "unit"))
    scale = RANGE_UNITS[unit]
    try:
        with operation_result_storage("auto", ram_budget_bytes=64 * 1024**2):
            for plan in request.plans:
                check_cancel(cancel)
                dataset = plan.dataset
                native = np.asarray(dataset.frequencies[plan.selection[2]], dtype=float)
                frequencies_hz = np.asarray(dataset._frequency_value_to_hz(native), dtype=float)
                problem = _uniform_problem(frequencies_hz)
                if problem:
                    result.skipped.append(f"{plan.name} ({problem})")
                    continue
                width = int(min(native.size, max(8, round(percent / 100.0 * native.size))))
                sweeps = SelectedSweeps(dataset, plan.selection)
                try:
                    ranges_m, starts, power = range_frequency_map(
                        sweeps, frequencies_hz, width=width, window=window, cancel=cancel)
                except ValueError as exc:
                    result.skipped.append(f"{plan.name} ({exc})")
                    continue
                centers = np.asarray([native[s:s + width].mean() for s in starts])
                # Reuse the owned destination: no second full native display
                # plane while converting from linear power to display units.
                if request.scale != "linear":
                    for rows, columns in result_blocks(power.shape):
                        check_cancel(cancel)
                        tile = np.asarray(power[rows, columns])
                        if np.any(np.isfinite(tile) & (tile <= 0)):
                            result.notes.append("Zero/nonpositive samples are omitted on logarithmic axes; positive levels are not floored.")
                        power[rows, columns] = dataset.linear_to_default_db(
                            np.where(np.isfinite(tile) & (tile > 0), tile, np.nan),
                            frequency_value=centers[rows, None], eps=0.)
                for block in result_blocks(power.shape):
                    power[block][~np.isfinite(power[block])] = np.nan
                x = common.values_for_display(request.reference, dataset, "frequency", centers)
                y = ranges_m * scale
                reduction_options = {} if cancel is None else {"cancel": lambda: check_cancel(cancel)}
                x_display, y_display, image, reduced = common.decimate_image(
                    x, y, power, **reduction_options)
                selected_band = common.values_for_display(request.reference, dataset, "frequency", native[[0, -1]])
                panel = dict(name=plan.name, pol=str(dataset.polarizations[plan.selection[3][0]]),
                    sweeps=sweeps.complete_count, x=x_display, y=y_display, image=image,
                    resolution=C0 / (2.0 * float(np.mean(np.diff(frequencies_hz))) * width) * scale,
                    width=width, count=native.size, selected_band=selected_band, snapshot=None)
                result.items.append(panel)
                if reduced:
                    check_cancel(cancel)
                    snapshot = common.NativeSamples.image(power)
                    panel["snapshot"] = (selected_band if x.size == 1 else common.coordinate_edges(x),
                                         common.coordinate_edges(y), snapshot)
                    result.notes.append("Image shows block maxima; native cell probes use original samples. Numerical exports use full data; figure exports use this display.")
            check_cancel(cancel)
            result.notes = list(dict.fromkeys(result.notes))
            return result
    except BaseException:
        result.close()
        raise


def render(self) -> None:
    self.last_plot_mode = "range_freq"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    if self._button_checked(self.btn_phase):
        self.status.showMessage("Range–Freq maps show levels. Turn off Phase to plot one.")
        return

    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    for values, axis in ((az_values, "azimuths"), (freq_values, "frequencies"),
                         (elev_values, "elevations")):
        if values.size == 0:
            self.status.showMessage(f"Select one or more {axis} to plot.")
            return
    polarization = self._single_selection_value(self.list_pol, "polarization")
    if polarization is None:
        return
    plans, skipped = _plan_series(
        self, reference, datasets, az_values, elev_values, freq_values, polarization
    )
    if not plans:
        self._show_plot_status("No compatible one-to-one coordinates for the selected plot.")
        return
    if len(plans) > common.MAX_WATERFALL_PANELS:
        self.status.showMessage(
            f"Range–Freq blocked: select at most {common.MAX_WATERFALL_PANELS} datasets."
        )
        return

    controls = getattr(self, "analysis_controls", None)
    percent = controls.range_subband_percent() if controls is not None else 25.0
    window = controls.range_window() if controls is not None else "Hanning"
    unit = controls.range_unit() if controls is not None else "in"
    request = PlotRequest("range_freq", reference,
        tuple(PlotPlan(name, dataset, selection) for name, dataset, selection in plans),
        scale=self._plot_scale_mode(), options={"percent": percent, "window": window, "unit": unit})

    def publish(result):
        for note in result.notes:
            self._note_plot_render(note)
        _display_panels(self, result.items, skipped + result.skipped, datasets, reference, unit)

    # Estimate transform work to schedule substantial narrow-band/many-sweep
    # jobs in the same cancellable, latest-request-wins runner as Waterfall.
    work = 0
    for plan in request.plans:
        count = len(plan.selection[2])
        if count >= 8:
            width = int(min(count, max(8, round(percent / 100.0 * count))))
            work += range_frequency_plan(len(plan.selection[0]) * len(plan.selection[1]), count, width)["transform_cells"]
    return self._submit_plot_preparation(request, lambda cancel: prepare_range_frequency(request, cancel),
                                  publish, cells=work)


def _display_panels(self, panels, skipped, datasets, reference, unit):
    if not panels:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No range–frequency map could be formed.{detail}")
        return

    self._remove_colorbar()
    self._release_native_plot_samples()
    self.plot_figure.clear()
    axes = self.plot_figure.subplots(nrows=len(panels), ncols=1, squeeze=False)[:, 0]
    self.plot_axes = list(axes)
    self.plot_ax = self.plot_axes[0]
    self.plot_figure._grim_line_plot_signature = None
    self.plot_figure.set_facecolor(self._current_plot_bg())
    zmin, zmax = self.spin_plot_zmin.value(), self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    limits = common.finite_data_limits(panel["image"] for panel in panels)
    vmin = zmin if use_clamp else (limits[0] if limits else None)
    vmax = zmax if use_clamp else (limits[1] if limits else None)
    if not use_clamp and limits and not self._plot_scale_is_linear():
        vmin = max(vmin, vmax - 60.0)  # 60 dB below the peak keeps sidelobes readable
    meshes = []
    for ax, panel in zip(self.plot_axes, panels):
        self._style_axes(ax)
        x, y = panel["x"], panel["y"]
        if len(x) == 1:
            # With 100% width (or only eight samples) this is one full-band
            # profile. Center-only pcolormesh coordinates give it zero width,
            # making a valid result invisible. Draw its selected band extent.
            x = panel["selected_band"]
            y = np.r_[y[0] - (y[1]-y[0])/2, (y[:-1]+y[1:])/2,
                      y[-1] + (y[-1]-y[-2])/2]
        mesh = ax.pcolormesh(
            x, y, panel["image"].T, shading="auto",
            cmap=self._effective_colormap(), vmin=vmin, vmax=vmax,
        )
        coordinates = mesh.get_coordinates()
        mesh._grim_rectilinear_data = (
            coordinates[0, :, 0], coordinates[:, 0, 1],
            mesh.get_array().reshape(panel["image"].T.shape),
        )
        self._attach_native_image_snapshot(mesh, panel["snapshot"])
        meshes.append(mesh)
        band_label = ("full selected band" if panel['width'] == panel['count']
                      else f"sub-band {panel['width']}/{panel['count']} samples")
        band_low, band_high = panel["selected_band"]
        ax.set_title(
            f"{panel['name']} | Pol {panel['pol']}, mean of {panel['sweeps']} sweeps\n"
            f"Selected {band_low:g}–{band_high:g} {common.axis_unit(reference, 'frequency')} | "
            f"{band_label} "
            f"({panel['resolution']:.3g} {unit} resolution)",
            color=self._current_plot_text(), fontsize=9,
        )
        frequency_label = self._plot_axis_label(reference, "frequency")
        ax.set_xlabel(("Selected band: " + frequency_label) if len(panel["x"]) == 1
                      else "Sub-band center " + frequency_label[:1].lower() + frequency_label[1:])
        ax.set_ylabel(f"Down range ({unit})")
    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(meshes[-1], ax=self.plot_axes)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        colorbar.set_label(
            self._display_axis_label(datasets, tag=" range profile"),
            color=self._current_plot_text(),
        )
        colorbar.ax.tick_params(colors=self._current_plot_text())

    x_all = np.concatenate([panel["selected_band"] if len(panel["x"]) == 1 else panel["x"]
                            for panel in panels])
    y_all = np.concatenate([panel["y"] for panel in panels])
    for spin, value in (
        (self.spin_plot_xmin, float(np.min(x_all))), (self.spin_plot_xmax, float(np.max(x_all))),
        (self.spin_plot_ymin, float(np.min(y_all))), (self.spin_plot_ymax, float(np.max(y_all))),
    ):
        spin.blockSignals(True)
        common.set_spin_value(spin, value)
        spin.blockSignals(False)
    self._apply_plot_limits()
    status = "Range–frequency map updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
