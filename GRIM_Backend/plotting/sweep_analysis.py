"""Shared, Qt-free sweep statistics for interactive plots and Python replay."""
from __future__ import annotations

import warnings
import numpy as np

from .modes import common


def options(values=None):
    values = dict(values or {})
    mode = str(values.get("mode", "median"))
    if mode not in {"individual", "mean", "median", "percentile", "max"}:
        raise ValueError(f"Unknown sweep aggregation {mode!r}")
    percentile = float(values.get("percentile", 90.))
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("Sweep percentile must be between 0 and 100")
    return {"mode": mode, "percentile": percentile}


def owner_options(owner):
    getter = getattr(getattr(owner, "analysis_controls", None), "sweep_aggregation", None)
    return options(getter() if callable(getter) else None)


def statistic_name(config, *, phase=False):
    mode, percentile = config["mode"], config["percentile"]
    if phase:
        return {"individual": "Individual phase", "mean": "Circular mean phase",
                "median": "P50 circular phase", "percentile": f"Phase of P{percentile:g} nearest-rank power sample",
                "max": "Phase of strongest return"}[mode]
    return {"individual": "Individual cut", "mean": "Mean linear power", "median": "P50",
            "percentile": f"P{percentile:g} linear power", "max": "Worst case (maximum power)"}[mode]


def short_statistic(config, *, phase=False):
    mode, percentile = config["mode"], config["percentile"]
    if phase:
        return {"individual": "Phase", "mean": "Circular mean", "median": "P50",
                "percentile": f"Phase@P{percentile:g} power", "max": "Phase@max power"}[mode]
    return {"individual": "Cut", "mean": "Mean power", "median": "P50",
            "percentile": f"P{percentile:g}", "max": "Max"}[mode]


def axis_tag(config, *, phase=False):
    return "" if config["mode"] == "individual" else " " + short_statistic(config, phase=phase)


def reduce_power(power, config):
    """Equal-sample reduction along azimuth; no mean of dB or coherent field."""
    power = np.asarray(power, dtype=float)
    valid = np.isfinite(power) & (power >= 0.)
    clean = np.where(valid, power, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mode = config["mode"]
        if mode == "mean":
            value = np.nanmean(clean, axis=0)
        elif mode == "max":
            value = np.nanmax(clean, axis=0)
        else:
            q = 50. if mode == "median" else config["percentile"]
            value = np.nanpercentile(clean, q, axis=0, method="linear")
    return value, np.sum(valid, axis=0)


def reduce_phase(field, config):
    """Circular reductions, or phase of a named power-ranked observation.

    Phase has no scalar worst-case or ordered percentile. For max/percentile
    we explicitly select the strongest/nearest-rank power observation and
    show its phase; the magnitude percentile itself still interpolates power.
    """
    field = np.asarray(field)
    power = np.abs(field) ** 2
    valid = np.isfinite(field) & (power > 0.)
    angles = np.where(valid, np.degrees(np.angle(field)), np.nan)
    count = np.sum(valid, axis=0)
    mode = config["mode"]
    if mode == "median":
        result = common.circular_median_degrees(angles, axis=0)
    elif mode == "mean":
        phasors = np.where(valid, np.exp(1j * np.deg2rad(angles)), 0.)
        total = np.sum(phasors, axis=0)
        result = np.where(np.abs(total) > 1.e-12 * np.maximum(count, 1), np.degrees(np.angle(total)), np.nan)
    else:
        order = np.argsort(np.where(valid, power, np.inf), axis=0, kind="stable")
        q = 100. if mode == "max" else config["percentile"]
        rank = np.clip((np.ceil(q / 100. * count) - 1).astype(int), 0, len(field) - 1)
        chosen = np.take_along_axis(order, rank[None, :], axis=0)[0]
        result = np.take_along_axis(angles, chosen[None, :], axis=0)[0]
        result = np.where(count > 0, result, np.nan)
    return result, count


def display_values(dataset, values, *, phase, scale, frequency, preserve_zeros=False):
    values = np.asarray(values, dtype=float)
    if phase:
        if str((dataset.units or {}).get("phase_wrap", "-180_180")) == "0_360":
            return np.mod(values, 360.)
        return np.mod(values + 180., 360.) - 180.
    if scale == "linear":
        return values
    return dataset.linear_to_default_db(np.where((values > 0.) | (preserve_zeros & (values == 0.)), values, np.nan), frequency_value=frequency, eps=0.)


def iter_series(name, dataset, reference, selection, *, sweep_axis, config=None, phase=False, scale="dbsm", preserve_zeros=False, cancel=None):
    """Yield x, y, informative label and semantic identity for each cut."""
    config = options(config)
    az_idx, el_idx, fr_idx, pol_idx = selection
    pol_idx = int(np.asarray(pol_idx).ravel()[0])
    fixed_axis = "elevation" if sweep_axis == "frequency" else "frequency"
    sweep_idx = fr_idx if sweep_axis == "frequency" else el_idx
    fixed_indices = el_idx if sweep_axis == "frequency" else fr_idx
    native_sweep = np.asarray(getattr(dataset, "frequencies" if sweep_axis == "frequency" else "elevations"))[sweep_idx]
    x = common.values_for_display(reference, dataset, sweep_axis, native_sweep)
    az_native = np.asarray(dataset.azimuths)[az_idx]
    az_display = common.values_for_display(reference, dataset, "azimuth", az_native)
    az_name, az_unit = common.angular_axis_name(reference, "azimuth"), common.axis_unit(reference, "azimuth")
    individual = config["mode"] == "individual"
    for fixed_index in fixed_indices:
        fixed_native = float(getattr(dataset, "elevations" if fixed_axis == "elevation" else "frequencies")[fixed_index])
        fixed_display = float(common.values_for_display(reference, dataset, fixed_axis, [fixed_native])[0])
        fixed_name = common.angular_axis_name(reference, "elevation") if fixed_axis == "elevation" else "Freq"
        fixed_name = "El" if fixed_name == "Elevation" else fixed_name
        frequency = native_sweep if sweep_axis == "frequency" else fixed_native
        # Yield individual cuts lazily so the existing visible-series limit
        # does not force calculation of every unused cut.
        az_groups = [(index, [native_index]) for index, native_index in enumerate(az_idx)] if individual else [(None, az_idx)]
        for az_position, group in az_groups:
            # Every reduced output coordinate is independent. Tile the sweep
            # dimension while retaining every azimuth contributing to its exact
            # statistic; this changes allocation size, never the population.
            from .requests import check_cancel
            width = max(1, common.REDUCTION_BLOCK_CELLS // max(1, len(group)))
            reduced, count = np.empty(len(sweep_idx)), np.empty(len(sweep_idx), dtype=np.intp)
            for start in range(0, len(sweep_idx), width):
                check_cancel(cancel)
                stop = min(len(sweep_idx), start + width)
                selected_sweep = np.asarray(sweep_idx)[start:stop]
                selection_key = np.ix_(group, [fixed_index], selected_sweep, [pol_idx]) if sweep_axis == "frequency" else np.ix_(group, selected_sweep, [fixed_index], [pol_idx])
                raw = dataset.rcs_slice(selection_key) if phase else dataset.power[selection_key]
                block = raw[:, 0, :, 0] if sweep_axis == "frequency" else raw[:, :, 0, 0]
                if individual:
                    if phase:
                        valid = np.isfinite(block[0]) & (np.abs(block[0]) > 0.)
                        values = np.where(valid, np.degrees(np.angle(block[0])), np.nan)
                    else:
                        valid = np.isfinite(block[0]) & (block[0] >= 0.)
                        values = np.where(valid, block[0], np.nan)
                    counts = valid.astype(int)
                else:
                    values, counts = reduce_phase(block, config) if phase else reduce_power(block, config)
                reduced[start:stop], count[start:stop] = values, counts
            if individual:
                description = f"{'Az' if az_name == 'Azimuth' else az_name} {az_display[az_position]:.6g} {az_unit}"
                az_identity = (float(az_native[az_position]),)
            else:
                description = short_statistic(config, phase=phase)
                az_identity = tuple(float(value) for value in az_native)
            display = display_values(dataset, reduced, phase=phase, scale=scale, frequency=frequency, preserve_zeros=preserve_zeros)
            low_count, high_count = int(np.min(count)), int(np.max(count))
            coverage = f"n{low_count}/{len(group)}" if low_count == high_count else f"n{low_count}–{high_count}/{len(group)}"
            label = (f"{name} | Pol {dataset.polarizations[pol_idx]}, {fixed_name} {fixed_display:.6g} "
                     f"{common.axis_unit(reference, fixed_axis)}, {description}; {coverage}")
            key = ("sweep", sweep_axis, config["mode"], config["percentile"], az_identity,
                   tuple(float(v) for v in native_sweep), fixed_native, str(dataset.polarizations[pol_idx]))
            yield np.asarray(x), np.asarray(display), label, key
