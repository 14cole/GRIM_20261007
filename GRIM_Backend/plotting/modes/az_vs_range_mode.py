"""Azimuth vs Down-Range image — partial ISAR.

For each selected azimuth, IFFT over frequency to build a range profile;
stack the profiles side by side. Unlike the ISAR mode this does NOT FFT
across azimuth, so the X axis stays in degrees rather than collapsing to
a spatial cross-range coordinate.

Useful for spotting which look-angles a particular scatterer lights up at,
diagnosing range-walk before doing a full ISAR, or quickly seeing target
extent without committing to a small azimuth window.
"""
from __future__ import annotations

import numpy as np

from GRIM_Backend.datasets.transforms import _declared_time_sign
from . import common
from .isar_mode import (
    _MAX_INTERP_COMPLEX_CELLS,
    _length_unit,
    _uniform_resample_plan,
    _unit_to_hz_scale,
    _angle_values_to_degrees,
    _ifft,
)
from GRIM_Backend.isar.interpolation import resample_pair


def _prepare_uniform_frequency_history(
    frequency_hz: np.ndarray,
    complex_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Gap-aware range-processing grid and per-row measurement weights.

    Interpolation is allowed only between nearby acquired frequencies. A large
    missing band and any interpolation touching an unknown complex sample stay
    zero-weighted rather than becoming apparently measured phase history.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    history = np.asarray(complex_history)
    if history.ndim != 2 or history.shape[1] != frequency_hz.size:
        raise ValueError(
            "azimuth/range phase history must have shape (azimuth, frequency)"
        )
    maximum_frequency_samples = max(
        frequency_hz.size,
        _MAX_INTERP_COMPLEX_CELLS // max(history.shape[0], 1),
    )
    plan = _uniform_resample_plan(
        frequency_hz,
        max_output_samples=maximum_frequency_samples,
    )
    finite = np.isfinite(history)
    clean = np.nan_to_num(
        history,
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.complex64, copy=False)
    uniform_history, interpolated_validity = resample_pair(
        frequency_hz, clean, finite.astype(np.float32), plan["target"],
        axis=1, support=plan["support"],
    )
    weights = (interpolated_validity >= 1.0 - 1.0e-6).astype(np.float32)
    uniform_history = np.asarray(uniform_history, dtype=np.complex64) * weights
    return (
        np.asarray(plan["target"], dtype=float),
        uniform_history,
        weights,
        dict(plan["info"]),
    )


def _form_range_image(history, sample_weights, window):
    """Single-precision FFT with double-precision coherent-gain accumulation."""
    taper = np.asarray(window, dtype=np.float32)
    weighted = np.asarray(history, dtype=np.complex64) * taper[None, :]
    weighted *= sample_weights
    image = np.fft.fftshift(_ifft(weighted, n=weighted.shape[1], axis=1), axes=1)
    gain = np.sum(sample_weights * taper[None, :], axis=1, dtype=np.float64) / weighted.shape[1]
    usable = gain > 0
    np.divide(image, gain[:, None], out=image, where=usable[:, None])
    image[~usable] = np.nan + 1j * np.nan
    return image, usable


def _range_display_values(dataset, magnitude: np.ndarray, *, linear: bool) -> np.ndarray:
    """Convert coherent range amplitude to generic image intensity.

    A frequency IFFT does not, by itself, preserve a calibrated physical RCS
    normalization. Keep the familiar amplitude-dB scaling without claiming
    the result is dBsm/dBke.
    """

    del dataset  # retained in the public helper signature for compatibility
    magnitude = np.asarray(magnitude, dtype=float)
    intensity = magnitude ** 2
    if linear:
        return intensity
    intensity = np.where(np.isfinite(intensity), intensity, np.nan)
    return 10.0 * np.log10(np.maximum(intensity, 1.0e-12))


def _range_display_grid(azimuths, ranges, image, *, azimuth_width, max_side):
    """Peak-pool with actual bin edges, leaving unmeasured angles blank.

    Each acquired angle occupies one native sampling-width cell centerd on
    that angle. Pooling never crosses an angular gap. Partial final blocks
    retain their real extent instead of stretching the image to fit.
    """
    azimuths = np.asarray(azimuths, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    width = float(azimuth_width)
    dy = float(ranges[1] - ranges[0])
    y_edges = np.r_[ranges - dy / 2.0, ranges[-1] + dy / 2.0]
    y_starts = np.arange(0, len(ranges), max(1, int(np.ceil(len(ranges) / max_side))))
    pooled = np.fmax.reduceat(image, y_starts, axis=0)
    y_edges = y_edges[np.r_[y_starts, len(ranges)]]
    breaks = np.r_[0, np.flatnonzero(np.diff(azimuths) > width * (1 + 1e-6)) + 1,
                   len(azimuths)]
    # Reserve room for one transparent column per gap in the display budget.
    max_blocks = max_side - (len(breaks) - 2)
    if len(breaks) - 1 > max_blocks:
        raise ValueError("too many disconnected azimuth samples; select fewer angles")
    stride = max(1, int(np.ceil(len(azimuths) / max_blocks)))
    while True:
        blocks = [(first, min(first + stride, end))
                  for start, end in zip(breaks[:-1], breaks[1:])
                  for first in range(start, end, stride)]
        if len(blocks) <= max_blocks:
            break
        stride *= 2
    edges = [float(azimuths[0] - width / 2)]
    columns = []
    for first, end in blocks:
        left = float(azimuths[first] - width / 2)
        if left > edges[-1] + width * 1e-6:
            columns.append(np.full(pooled.shape[0], np.nan))
            edges.append(left)
        columns.append(np.fmax.reduce(pooled[:, first:end], axis=1))
        edges.append(float(azimuths[end - 1] + width / 2))
    return np.asarray(edges), y_edges, np.column_stack(columns), (
        len(y_starts) < len(ranges) or len(blocks) < len(azimuths)
    )


def render(self) -> None:
    self.last_plot_mode = "az_vs_range"
    self._start_plot_render()
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets([("Dataset", self.active_dataset)])
    if reference is None:
        return
    try:
        time_sign = _declared_time_sign(self.active_dataset)
    except ValueError as exc:
        self.status.showMessage(f"Az vs Down-Range blocked: {exc}")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    aperture = getattr(self, "chk_isar_aperture", None)
    if aperture is not None and aperture.isChecked() and az_indices:
        center, width = float(self.spin_isar_ap_center.value()), float(self.spin_isar_ap_width.value())
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            self.status.showMessage("Aperture center/width must be finite and width positive.")
            return
        degrees = _angle_values_to_degrees(self.active_dataset, 'azimuth', self.active_dataset.azimuths[az_indices])
        az_indices = [i for i, distance in zip(az_indices, abs((degrees-center+180) % 360-180)) if distance <= width/2+1e-9]
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    band = getattr(self, "chk_isar_freq_band", None)
    if band is not None and band.isChecked():
        lo, hi = float(self.spin_isar_freq_min.value()), float(self.spin_isar_freq_max.value())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            self.status.showMessage("Frequency band max must exceed min; both must be finite.")
            return
        freq_indices = [i for i in freq_indices if lo <= self.active_dataset.frequencies[i] <= hi]
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for range processing.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    # Sort axes ascending; build the (n_az, n_freq) complex slice.
    az_values = self.active_dataset.azimuths[az_indices].astype(float)
    az_order = np.argsort(az_values)
    sorted_az_indices = [az_indices[i] for i in az_order]
    az_values = az_values[az_order]
    if not np.all(np.isfinite(az_values)) or np.any(np.diff(az_values) <= 0):
        self.status.showMessage(
            f"{self._plot_axis_name(reference, 'azimuth')} samples must be strictly increasing."
        )
        return

    freq_values = self.active_dataset.frequencies[freq_indices].astype(float)
    freq_order = np.argsort(freq_values)
    sorted_freq_indices = [freq_indices[i] for i in freq_order]
    freq_values = freq_values[freq_order]
    if np.any(np.diff(freq_values) <= 0):
        self.status.showMessage("Frequency samples must be strictly increasing.")
        return

    rcs_slice = self.active_dataset.rcs_slice(
        np.ix_(sorted_az_indices, [elev_idx], sorted_freq_indices, [pol_idx])
    )[:, 0, :, 0]
    if time_sign == "-jwt":
        # Work in the same +jwt convention used by the range-frequency map.
        rcs_slice = np.conj(rcs_slice)
    if not np.any(np.isfinite(rcs_slice)):
        self.status.showMessage(
            "No compatible phase-aware data for the selected azimuth, "
            "elevation, frequency, and polarization values."
        )
        return
    # IFFT along frequency requires a uniform grid. Preserve missing bands and
    # unknown phase as zero-weight observations; never bridge them as data.
    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    try:
        freq_hz_uniform, rcs_slice, sample_weights, frequency_sampling = (
            _prepare_uniform_frequency_history(freq_hz, rcs_slice)
        )
    except ValueError as exc:
        self.status.showMessage(f"Az vs Down-Range blocked: {exc}")
        return
    fr_nonuniformity = float(frequency_sampling["non_uniformity"])
    n_freq = freq_hz_uniform.size
    df = float(np.mean(np.diff(freq_hz_uniform)))

    # Window over freq (re-uses ISAR window selector).
    win_freq = self._isar_window(n_freq)
    range_image, usable_rows = _form_range_image(rcs_slice, sample_weights, win_freq)
    if not np.any(usable_rows):
        self.status.showMessage(
            "No azimuth row has enough finite, supported phase history for range processing."
        )
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(
        units_combo.currentText() if units_combo else "in"
    )
    c0 = 299_792_458.0
    range_axis = (
        np.fft.fftshift(np.fft.fftfreq(n_freq, d=df)) * (c0 / 2.0) * unit_scale
    )

    magnitude = np.abs(range_image)

    # Optional peak normalisation (re-uses ISAR toggle).
    pn_widget = getattr(self, "chk_isar_peak_normalize", None)
    peak_norm = bool(pn_widget.isChecked()) if pn_widget else False
    if peak_norm:
        peak = float(np.nanmax(magnitude))
        if peak > 0.0:
            magnitude = magnitude / peak

    display = _range_display_values(
        self.active_dataset,
        magnitude,
        linear=self._plot_scale_is_linear(),
    )
    max_side = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
    native_steps = np.diff(np.sort(np.unique(self.active_dataset.azimuths)))
    width = float(np.min(native_steps)) if native_steps.size else (
        float(np.deg2rad(1.0)) if self._plot_axis_unit(reference, "azimuth") == "rad" else 1.0
    )
    try:
        x_edges, y_edges, display_for_plot, decimated = _range_display_grid(
            az_values, range_axis, display.T, azimuth_width=width, max_side=max_side
        )
    except ValueError as exc:
        self.status.showMessage(f"Az vs Down-Range blocked: {exc}")
        return
    if decimated:
        self._note_plot_render(
            "Large range image was peak-preserving display-decimated for responsive "
            "interaction; narrow the selected axes for full display resolution."
        )

    # Build the figure.
    self._remove_colorbar()
    self.plot_figure.clear()
    self.plot_ax = self.plot_figure.add_subplot(111)
    self.plot_axes = None
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    color_options = dict(cmap=cmap, vmin=zmin if use_clamp else None,
                         vmax=zmax if use_clamp else None)
    uniform_edges = all(np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-6, atol=0)
                        for edges in (x_edges, y_edges))
    if uniform_edges:
        mesh = self.plot_ax.imshow(
            display_for_plot,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower", aspect="auto", interpolation="nearest", **color_options,
        )
    else:
        mesh = self.plot_ax.pcolormesh(
            x_edges, y_edges, display_for_plot, shading="flat", rasterized=True,
            **color_options,
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(f"Down-Range ({unit_name})")
    elev_value = self.active_dataset.elevations[elev_idx]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    pol_value = self.active_dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        f"{self._plot_axis_name(reference, 'azimuth')} vs Down-Range | "
        f"{elev_name} {elev_value:g} {elev_unit} | Pol {pol_value}",
        color=self._current_plot_text(),
    )

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label(
                "Range image intensity (linear, a.u.)",
                color=self._current_plot_text(),
            )
        else:
            colorbar.set_label(
                "Range image intensity (dB re 1 a.u.)",
                color=self._current_plot_text(),
            )
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    # Update axis spinboxes to match the new view.
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(x_edges[0]))
    self.spin_plot_xmax.setValue(float(x_edges[-1]))
    self.spin_plot_ymin.setValue(float(y_edges[0]))
    self.spin_plot_ymax.setValue(float(y_edges[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()

    note = ""
    if fr_nonuniformity >= 1e-3:
        note = f" — resampled frequency (Δ-spread {fr_nonuniformity*100:.1f}%)"
    gap_count = int(frequency_sampling.get("gap_count", 0))
    if gap_count:
        unsupported = 100.0 * float(
            frequency_sampling.get("unsupported_fraction", 0.0)
        )
        note += (
            f" — {gap_count} missing frequency band(s) kept zero-weighted "
            f"({unsupported:.1f}% unsupported grid)"
        )
    self._show_plot_status(f"Az vs Down-Range updated{note}.")
