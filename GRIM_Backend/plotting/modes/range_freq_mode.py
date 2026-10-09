"""Down-range profiles over sliding frequency sub-bands.

Each column is the range profile of one sub-band: the selected complex
sweeps are windowed, inverse-FFT'd with zero padding, and their power is
averaged over the selected azimuths and elevations. A point scatterer stays at
one range in every sub-band; cavities, travelling waves, and resonances drift
in range or appear only in part of the band. Profiles are scaled by the
window's coherent gain, so an isolated point shows its own level at its peak.
"""

from __future__ import annotations

import numpy as np

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.transforms import _declared_time_sign

from . import common
from .azimuth_rect_mode import _plan_series
from .isar_mode import _window_array

MAX_COLUMNS = 200
MAX_WORK_CELLS = 64_000_000
RANGE_UNITS = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "in": 1.0 / 0.0254, "ft": 1.0 / 0.3048}


def subband_starts(count: int, width: int, max_columns: int = MAX_COLUMNS) -> list[int]:
    """Bounded sub-band starts including both ends of the selected sweep."""
    if not 1 <= width <= count or max_columns < 1:
        raise ValueError("Sub-band width and column limit must be positive and fit the selection.")
    positions = count - width + 1
    if positions > 1 and max_columns < 2:
        raise ValueError("At least two columns are needed to cover both ends of the selection.")
    return np.linspace(0, positions - 1, min(positions, max_columns), dtype=int).tolist()


def range_frequency_map(sweeps, frequencies_hz, *, width: int, window: str):
    """Return ``(ranges_m, starts, power)`` with power shaped (sub-bands, ranges).

    ``sweeps`` are complex (n_sweeps, n_frequencies) arrays on a uniform
    frequency grid in GRIM's exp(+jwt) law; incomplete sweeps must already
    be removed.
    """
    sweeps = np.asarray(sweeps, dtype=np.complex128)
    count = frequencies_hz.size
    step_hz = float(np.mean(np.diff(frequencies_hz)))
    taper = _window_array(window, width)
    gain = float(np.sum(taper))
    size = max(64, 1 << int(np.ceil(np.log2(4 * width))))
    starts = subband_starts(count, width)
    power = np.empty((len(starts), size))
    for column, start in enumerate(starts):
        profile = np.fft.ifft(sweeps[:, start:start + width] * taper, n=size, axis=-1)
        power[column] = np.mean(np.abs(profile * (size / gain)) ** 2, axis=0)
    ranges = np.fft.fftshift(np.fft.fftfreq(size, d=step_hz)) * (C0 / 2.0)
    return ranges, starts, np.fft.fftshift(power, axes=1)


def _uniform_problem(frequencies_hz) -> str:
    if frequencies_hz.size < 8:
        return "needs at least 8 selected frequencies"
    steps = np.diff(frequencies_hz)
    step = float(np.mean(steps))
    if step <= 0.0 or np.max(np.abs(steps - step)) > 1.0e-6 * step:
        return "needs uniformly spaced selected frequencies"
    return ""


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
    scale = RANGE_UNITS[unit]

    panels = []
    for name, dataset, selection in plans:
        az_indices, elev_indices, freq_indices, pol_indices = selection
        native = np.asarray(dataset.frequencies[freq_indices], dtype=float)
        frequencies_hz = np.asarray(dataset._frequency_value_to_hz(native), dtype=float)
        problem = _uniform_problem(frequencies_hz)
        if problem:
            skipped.append(f"{name} ({problem})")
            continue
        width = int(min(native.size, max(8, round(percent / 100.0 * native.size))))
        field = dataset.rcs_slice(
            np.ix_(az_indices, elev_indices, freq_indices, [pol_indices[0]])
        )[..., 0].reshape(-1, native.size)
        if _declared_time_sign(dataset) == "-jwt":
            field = np.conj(field)
        sweeps = field[np.all(np.isfinite(field), axis=-1)]
        if sweeps.shape[0] == 0:
            skipped.append(f"{name} (no complete complex sweep; needs phase)")
            continue
        starts = subband_starts(native.size, width)
        size = max(64, 1 << int(np.ceil(np.log2(4 * width))))
        work = sweeps.shape[0] * len(starts) * size
        if work > MAX_WORK_CELLS:
            self.status.showMessage(
                f"Range–Freq blocked: {work:,} transform cells exceed the "
                f"{MAX_WORK_CELLS:,} interactive limit. Select fewer azimuths or "
                "elevations, or narrow the sub-band."
            )
            return
        ranges_m, starts, power = range_frequency_map(
            sweeps, frequencies_hz, width=width, window=window
        )
        centers = np.asarray([native[s:s + width].mean() for s in starts])
        display = self._display_from_linear(
            dataset, power, frequency_value=centers[:, None]
        )
        display = np.where(np.isfinite(display), display, np.nan)
        x_display = self._plot_axis_values(reference, dataset, "frequency", centers)
        x_display, y_display, image = self._bounded_plot_image(
            x_display, ranges_m * scale, display
        )
        step_hz = float(np.mean(np.diff(frequencies_hz)))
        panels.append({
            "name": name,
            "pol": str(dataset.polarizations[pol_indices[0]]),
            "sweeps": int(sweeps.shape[0]),
            "x": x_display,
            "y": y_display,
            "image": image,
            "resolution": C0 / (2.0 * step_hz * width) * scale,
            "width": width,
            "count": native.size,
            "selected_band": self._plot_axis_values(reference, dataset, "frequency", native[[0, -1]]),
        })

    if not panels:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No range–frequency map could be formed.{detail}")
        return

    self._remove_colorbar()
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
