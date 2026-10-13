"""Native-sample measurements and complex RF diagnostics, independent of Qt.

Phase-slope convention: tau = -dphi/df / (2*pi) for exp(+jwt), with the
opposite sign for exp(-jwt). See Keysight's Group Delay Format documentation.
Unwrapping is local to valid runs; no phase is inferred across missing samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import numpy as np
from scipy.signal import find_peaks

from GRIM_Backend.plotting.modes.common import axis_unit, convert_axis_values


@dataclass
class EngineeringCut:
    x: np.ndarray
    power: np.ndarray
    field: np.ndarray
    db: np.ndarray
    axis: str
    x_unit: str
    level_unit: str
    label: str
    time_convention: str
    notes: list[str]
    linear_quantity: str = ""
    polarization: str = ""
    fixed_coordinates: dict = dataclass_field(default_factory=dict)
    native_indices: np.ndarray | None = None


def dataset_cut(dataset, axis="frequency", *, azimuth_index=0, elevation_index=0,
                frequency_index=0, polarization="HH", label="Dataset"):
    """Read one physical cut, preferring authoritative raw complex samples."""
    numeric_axes = ("azimuth", "elevation", "frequency")
    if axis not in numeric_axes:
        raise ValueError("axis must be azimuth, elevation, or frequency")
    labels = [str(value).upper() for value in dataset.polarizations]
    pol_index = labels.index(str(polarization).upper())
    indices = [int(azimuth_index), int(elevation_index), int(frequency_index), pol_index]
    indices[numeric_axes.index(axis)] = slice(None)
    selection = tuple(indices)
    power = np.array(dataset.rcs_power[selection], dtype=float, copy=True).reshape(-1)
    field = np.array(dataset.rcs_slice(selection), dtype=complex, copy=True).reshape(-1)
    # In particular, sigma_2d raw amplitudes must first receive their native
    # frequency-dependent normalization; rcs_slice owns that contract.
    authoritative = np.isfinite(field.real) & np.isfinite(field.imag)
    power[authoritative] = np.abs(field[authoritative]) ** 2
    x = np.array(dataset.get_axis(axis), dtype=float, copy=True)
    frequencies = x if axis == "frequency" else float(dataset.frequencies[frequency_index])
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.asarray(dataset.linear_to_default_db(power, frequencies, eps=0.), dtype=float)
    declaration = dataset._declared_scalar_metadata("time_convention")
    notes = []
    for key in ("time_convention", "phase_reference", "polarization_basis"):
        evidence = dataset.inspect_scalar_metadata(key)
        values = tuple(evidence.declarations)
        if evidence.malformed_sources:
            notes.append(f"{key}: malformed declarations in {', '.join(evidence.malformed_sources)}")
        if not values:
            notes.append(f"{key}: unspecified")
        elif evidence.conflicting:
            notes.append(f"{key}: conflicting declarations {values}")
        else:
            notes.append(f"{key}: {values[0]}")
    fixed = []
    fixed_coordinates = {}
    for name, index in zip(numeric_axes, indices[:3]):
        if name != axis:
            fixed_coordinates[name] = (float(dataset.get_axis(name)[index]), axis_unit(dataset, name))
            fixed.append(f"{name}={float(dataset.get_axis(name)[index]):.9g} {axis_unit(dataset, name)}")
    return EngineeringCut(x, power, field, db, axis, axis_unit(dataset, axis),
                          dataset.default_log_unit(),
                          f"{label} | {polarization} | " + ", ".join(fixed),
                          declaration, notes, dataset.linear_quantity(), str(polarization), fixed_coordinates,
                          np.arange(len(x), dtype=int))


def finite_runs(x, valid, *, maximum_gap=None):
    """Index arrays for adjacent finite, increasing samples without bridging gaps."""
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if maximum_gap is not None and (not np.isfinite(maximum_gap) or maximum_gap <= 0):
        raise ValueError("maximum gap must be positive and finite")
    start = None
    for index in range(len(x)):
        boundary = index > 0 and (x[index] <= x[index - 1] or (
            maximum_gap is not None and x[index] - x[index - 1] > maximum_gap))
        if start is not None and (not valid[index] or boundary):
            yield np.arange(start, index)
            start = None
        if valid[index] and start is None:
            start = index
    if start is not None:
        yield np.arange(start, len(x))


def level_crossings(x, y, level, *, maximum_gap=None):
    """Interpolated crossing coordinates, separately within each finite run."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    crossings = []
    for run in finite_runs(x, np.isfinite(y), maximum_gap=maximum_gap):
        xr, yr = x[run], y[run]
        for i in range(len(run)):
            if yr[i] == level:
                # A plateau at the crossing level reports its two boundaries.
                if i == 0 or yr[i - 1] != level or i == len(run) - 1 or yr[i + 1] != level:
                    crossings.append(float(xr[i]))
            if i + 1 < len(run) and (yr[i] < level < yr[i + 1] or yr[i + 1] < level < yr[i]):
                crossings.append(float(xr[i] + (level - yr[i]) * (xr[i + 1] - xr[i]) / (yr[i + 1] - yr[i])))
    return np.asarray(sorted(set(crossings)), dtype=float)


def measure_trace(x, y, *, region=None, level=None, drop=3., maximum_gap=None):
    """Native sample extrema, local peaks, crossings, and peak-lobe width.

    ``drop`` uses the supplied ordinate unit. For dB traces 3 means a 3-dB
    drop, not half of a linear amplitude. Statistics describe the supplied
    displayed values and do not silently convert a dB average to power mean.
    """
    x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("trace coordinates and values must have equal lengths")
    if not np.isfinite(drop) or drop < 0:
        raise ValueError("peak drop must be finite and nonnegative")
    selected = np.isfinite(x)
    if region is not None:
        lo, hi = sorted(map(float, region))
        selected &= (x >= lo) & (x <= hi)
    valid = selected & np.isfinite(y)
    indices = np.flatnonzero(valid)
    out = {"selected_count": int(np.count_nonzero(selected)), "finite_count": len(indices),
           "coverage": float(len(indices) / max(1, np.count_nonzero(selected))),
           "peaks": [], "crossings": [], "bandwidth": None}
    if not len(indices):
        out["note"] = "No finite samples in the selected region."
        return out
    i_peak, i_min = int(indices[np.argmax(y[indices])]), int(indices[np.argmin(y[indices])])
    out.update(peak={"index": i_peak, "x": float(x[i_peak]), "y": float(y[i_peak])},
               minimum={"index": i_min, "x": float(x[i_min]), "y": float(y[i_min])},
               mean=float(np.mean(y[indices])), median=float(np.median(y[indices])),
               std=float(np.std(y[indices])), peak_to_peak=float(np.ptp(y[indices])))
    for run in finite_runs(x, valid, maximum_gap=maximum_gap):
        peaks, _ = find_peaks(y[run], plateau_size=True)
        out["peaks"].extend({"index": int(run[i]), "x": float(x[run[i]]), "y": float(y[run[i]])} for i in peaks)
        if i_peak in run:
            target = float(y[i_peak] - drop)
            if drop == 0:
                local = int(np.flatnonzero(run == i_peak)[0])
                lo = hi = local
                while lo > 0 and y[run[lo - 1]] == target: lo -= 1
                while hi + 1 < len(run) and y[run[hi + 1]] == target: hi += 1
                out["bandwidth_level"] = target
                out["bandwidth"] = {"left": float(x[run[lo]]), "right": float(x[run[hi]]),
                                    "width": float(x[run[hi]] - x[run[lo]])}
                continue
            crossing = level_crossings(x[run], y[run], target, maximum_gap=maximum_gap)
            left, right = crossing[crossing < x[i_peak]], crossing[crossing > x[i_peak]]
            out["bandwidth_level"] = target
            if len(left) and len(right):
                out["bandwidth"] = {"left": float(left[-1]), "right": float(right[0]),
                                    "width": float(right[0] - left[-1])}
            else:
                out["bandwidth_note"] = "Peak lobe is not bracketed by two crossings in one finite run."
    if level is not None:
        out["crossings"] = level_crossings(x, np.where(valid, y, np.nan), float(level), maximum_gap=maximum_gap).tolist()
    return out


def phase_trace(x, field, *, floor_db=-60., maximum_gap=None, region=None):
    """Mask weak/unknown fields and unwrap each finite run independently."""
    x, field = np.asarray(x, dtype=float), np.asarray(field, dtype=complex)
    magnitude = np.abs(field)
    valid = np.isfinite(x) & np.isfinite(magnitude) & (magnitude > 0)
    if region is not None:
        lo, hi = sorted(map(float, region))
        valid &= (x >= lo) & (x <= hi)
    peak = float(np.max(magnitude[valid])) if np.any(valid) else 0.
    if floor_db is not None and peak > 0:
        valid &= magnitude >= peak * 10. ** (float(floor_db) / 20.)
    wrapped = np.full(x.shape, np.nan)
    wrapped[valid] = np.angle(field[valid])
    unwrapped = np.full(x.shape, np.nan)
    runs = list(finite_runs(x, valid, maximum_gap=maximum_gap))
    for run in runs:
        unwrapped[run] = np.unwrap(wrapped[run])
    return {"wrapped": wrapped, "unwrapped": unwrapped, "valid": valid, "runs": runs,
            "finite_count": int(np.count_nonzero(valid)), "total_count": len(x)}


def phase_slope_fit(frequency_hz, field, *, floor_db=-60., region_hz=None,
                    maximum_gap_hz=None, time_sign="+jwt"):
    """Separate centered least-squares fits; never combine unknown phase branches."""
    if time_sign not in {"+jwt", "-jwt"}:
        raise ValueError("choose +jwt or -jwt to interpret a fitted delay")
    frequency = np.asarray(frequency_hz, dtype=float)
    order = np.argsort(frequency, kind="stable")
    frequency, field = frequency[order], np.asarray(field, dtype=complex)[order]
    phase = phase_trace(frequency, field, floor_db=floor_db,
                        maximum_gap=maximum_gap_hz, region=region_hz)
    fitted = np.full(frequency.shape, np.nan)
    residual = np.full(frequency.shape, np.nan)
    fits = []
    for run_number, run in enumerate(phase["runs"], start=1):
        row = {"run": run_number, "samples": len(run), "start_hz": float(frequency[run[0]]),
               "stop_hz": float(frequency[run[-1]])}
        if len(run) < 3:
            row["note"] = "At least three valid frequency samples are needed for a slope diagnostic."
            fits.append(row)
            continue
        x, y = frequency[run], phase["unwrapped"][run]
        centered = x - np.mean(x)
        denominator = float(np.dot(centered, centered))
        if denominator == 0:
            row["note"] = "No frequency span."
            fits.append(row)
            continue
        slope = float(np.dot(centered, y - np.mean(y)) / denominator)
        prediction = np.mean(y) + slope * centered
        error = y - prediction
        sse, sst = float(np.dot(error, error)), float(np.sum((y - np.mean(y)) ** 2))
        fitted[run], residual[run] = prediction, error
        df = np.diff(x)
        regular = np.allclose(df, np.median(df), rtol=1e-6, atol=0)
        row.update(slope_rad_per_hz=slope,
                   apparent_delay_s=(-1 if time_sign == "+jwt" else 1) * slope / (2 * np.pi),
                   residual_rms_deg=float(np.rad2deg(np.sqrt(sse / len(run)))),
                   residual_max_deg=float(np.rad2deg(np.max(np.abs(error)))),
                   r_squared=(1. - sse / sst) if sst > 0 else (1. if sse == 0 else None),
                   alias_period_s=float(1 / np.median(df)) if regular else None)
        fits.append(row)
    return {**phase, "x_hz": frequency, "sort_order": order, "fitted": fitted,
            "residual": residual, "fits": fits, "time_sign": time_sign,
            "note": "Apparent phase-slope delay, not target range or propagation proof. "
                    "Unwrapping assumes adjacent true phase changes below 180 degrees; aliasing cannot be excluded from these samples."}


def aligned_cuts(left, right):
    """Match physical coordinates without interpolation or reused source samples."""
    if left.axis != right.axis:
        raise ValueError("cut axes differ")
    converted = convert_axis_values(right.x, right.axis, right.x_unit, left.x_unit)
    ia, ib = [], []
    finite_a, finite_b = np.flatnonzero(np.isfinite(left.x)), np.flatnonzero(np.isfinite(converted))
    order_a = finite_a[np.argsort(left.x[finite_a])]
    order_b = finite_b[np.argsort(converted[finite_b])]
    a, b = left.x[order_a], converted[order_b]
    if not len(a) or not len(b):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), np.finfo(float).tiny)
    tol = np.finfo(float).eps * scale * 32
    j = 0
    for i, value in enumerate(a):
        while j < len(b) and b[j] < value - tol:
            j += 1
        if j < len(b) and abs(b[j] - value) <= tol:
            ia.append(int(order_a[i])); ib.append(int(order_b[j])); j += 1
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def phase_residual(left, right, *, floor_db=-60., maximum_gap=None, region=None):
    ia, ib = aligned_cuts(left, right)
    x, a, b = left.x[ia], left.field[ia], right.field[ib]
    va = phase_trace(x, a, floor_db=floor_db, maximum_gap=maximum_gap, region=region)["valid"]
    vb = phase_trace(x, b, floor_db=floor_db, maximum_gap=maximum_gap, region=region)["valid"]
    good = va & vb
    difference = np.full(x.shape, np.nan)
    difference[good] = np.angle((a[good] / np.abs(a[good])) * np.conj(b[good] / np.abs(b[good])))
    valid_values = difference[good]
    circular = np.mean(np.exp(1j * valid_values)) if len(valid_values) else 0j
    return {"x": x, "phase_difference_rad": difference, "common_count": len(x),
            "valid_count": int(np.count_nonzero(good)),
            "circular_mean_deg": float(np.rad2deg(np.angle(circular))) if abs(circular) > 1e-12 else None,
            "rms_wrapped_deg": float(np.sqrt(np.mean(np.rad2deg(valid_values) ** 2))) if len(valid_values) else None}


def channel_comparison(left, right, *, floor_db=-60., region=None):
    """Channel power ratio and phase difference on common native coordinates."""
    ia, ib = aligned_cuts(left, right)
    x, pa, pb = left.x[ia], left.power[ia], right.power[ib]
    valid = np.isfinite(pa) & np.isfinite(pb) & (pa >= 0) & (pb > 0)
    selected = np.ones(x.shape, dtype=bool)
    if region is not None:
        lo, hi = sorted(map(float, region))
        selected &= (x >= lo) & (x <= hi)
        valid &= selected
    for power in (pa, pb):
        positive = power[selected & np.isfinite(power) & (power > 0)]
        if floor_db is not None and positive.size:
            valid &= power >= np.max(positive) * 10. ** (float(floor_db) / 10.)
    ratio_db = np.full(x.shape, np.nan)
    with np.errstate(divide="ignore"):
        ratio_db[valid] = 10. * np.log10(pa[valid]) - 10. * np.log10(pb[valid])
    residual = phase_residual(left, right, floor_db=floor_db, region=region)
    good = np.isfinite(residual["phase_difference_rad"]) & valid
    a, b = left.field[ia], right.field[ib]
    scale = max(float(np.max(np.abs(a[good]))) if np.any(good) else 0.,
                float(np.max(np.abs(b[good]))) if np.any(good) else 0.)
    relative_error = None
    if scale > 0:
        aa, bb = a[good] / scale, b[good] / scale
        denom = .5 * np.sum(np.abs(aa) ** 2 + np.abs(bb) ** 2)
        relative_error = float(np.sqrt(np.sum(np.abs(aa - bb) ** 2) / denom)) if denom else None
    return {**residual, "ratio_db": ratio_db,
            "finite_ratio_count": int(np.count_nonzero(np.isfinite(ratio_db))),
            "normalized_complex_mismatch": relative_error}
