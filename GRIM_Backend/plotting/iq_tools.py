"""Native-sample complex-field diagnostics, without plotting or Qt dependencies.

I and Q retain the field units whose squared magnitude is the stored linear
quantity. Display normalization never changes the reported native field or dB
level. Complex comparisons match physical coordinates without interpolation.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.plotting.engineering_tools import aligned_cuts, phase_trace
from GRIM_Backend.plotting.modes.common import convert_axis_values


def _quantity(cut):
    declared = str(getattr(cut, "linear_quantity", "") or "").strip().lower()
    if declared:
        return "ratio" if declared == "power_ratio" else declared
    return {"dbsm": "sigma_3d", "dbke": "sigma_2d", "db": "ratio"}.get(
        str(cut.level_unit).strip().lower(), "unknown")


def field_unit(cut):
    """Units of each component, not units of squared magnitude or RF volts."""
    return {"sigma_3d": "m", "sigma_2d": "sqrt(m)", "ratio": "dimensionless"}.get(
        _quantity(cut), "native field units")


def _scaling_name(value):
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if value not in {"native", "common_peak", "phase_only"}:
        raise ValueError("choose native, common_peak, or phase_only scaling")
    return value


def _selected(x, region):
    selected = np.isfinite(x)
    if region is not None:
        lo, hi = sorted(map(float, region))
        selected &= (x >= lo) & (x <= hi)
    return selected


def _finite(field):
    return np.isfinite(field.real) & np.isfinite(field.imag)


def _native_indices(cut):
    source = getattr(cut, "native_indices", None)
    if source is None:
        return np.arange(len(cut.x), dtype=int)
    source = np.array(source, dtype=int, copy=True)
    if source.shape != np.shape(cut.x):
        raise ValueError("native indices must have one entry per cut sample")
    return source


def _peak(fields):
    peaks = [float(np.max(np.abs(values[_finite(values)]))) for values in fields
             if np.any(_finite(values))]
    return max(peaks, default=0.)


def _display(field, scaling, common_peak):
    """Return a copy; use one scalar for every field in a comparison."""
    result = np.array(field, dtype=complex, copy=True)
    if scaling == "phase_only":
        magnitude = np.abs(result)
        good = _finite(result) & (magnitude > 0)
        result[good] /= magnitude[good]
        result[~good] = complex(np.nan, np.nan)
        return result, None
    scale = 1.
    if scaling == "common_peak":
        scale = _peak([result]) if common_peak is None else float(common_peak)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("common peak must be finite and nonnegative")
        # An all-zero collection remains at the origin rather than dividing by 0.
        scale = scale or 1.
        result /= scale
    return result, scale


def _runs(x, valid, *, maximum_gap=None, source_indices=()):
    """Connected native samples; do not join across unmatched A or B samples."""
    if maximum_gap is not None and (not np.isfinite(maximum_gap) or maximum_gap <= 0):
        raise ValueError("maximum gap must be positive and finite")
    start = None
    out = []
    for index in range(len(x)):
        boundary = index > 0 and (
            x[index] == x[index - 1]
            or (maximum_gap is not None and abs(x[index] - x[index - 1]) > maximum_gap)
            or any(abs(int(indices[index]) - int(indices[index - 1])) != 1
                   for indices in source_indices))
        if start is not None and (not valid[index] or boundary):
            out.append(np.arange(start, index, dtype=int))
            start = None
        if valid[index] and start is None:
            start = index
    if start is not None:
        out.append(np.arange(start, len(x), dtype=int))
    return out


def iq_series(cut, *, scaling="native", common_peak=None, floor_db=-60.,
              maximum_gap=None, region=None):
    """Describe an I/Q trajectory while retaining exact native readout values.

    Weak finite samples and exact zeros remain visible in native/common-peak
    trajectories. ``phase_valid`` separately identifies reliable phase under
    the chosen relative floor; zero has no phase, including phase-only mode.
    """
    scaling = _scaling_name(scaling)
    x = np.array(cut.x, dtype=float, copy=True)
    native = np.array(cut.field, dtype=complex, copy=True)
    db = np.array(cut.db, dtype=float, copy=True)
    if x.ndim != 1 or native.shape != x.shape or db.shape != x.shape:
        raise ValueError("cut coordinates, field, and levels must be equal-length vectors")
    selected = _selected(x, region)
    normalization = common_peak
    if scaling == "common_peak" and normalization is None:
        normalization = _peak([native[selected]])
    display, scale = _display(native, scaling, normalization)
    valid = selected & _finite(display)
    phase = phase_trace(x, native, floor_db=floor_db, maximum_gap=maximum_gap, region=region)
    native_indices = _native_indices(cut)
    return {"x": x, "field": display, "native_field": native, "db": db,
            "phase_deg": np.rad2deg(phase["wrapped"]), "valid": valid,
            "phase_valid": phase["valid"],
            "runs": _runs(x, valid, maximum_gap=maximum_gap, source_indices=(native_indices,)),
            "native_indices": native_indices, "field_unit": field_unit(cut),
            "display_unit": field_unit(cut) if scaling == "native" else "dimensionless",
            "scaling": scaling, "scale": scale, "notes": list(cut.notes),
            "axis": cut.axis, "x_unit": cut.x_unit, "level_unit": cut.level_unit,
            "linear_quantity": _quantity(cut),
            "polarization": str(getattr(cut, "polarization", "") or ""),
            "fixed_coordinates": dict(getattr(cut, "fixed_coordinates", {}) or {}),
            "label": cut.label}


def _compatible(left, right):
    if _quantity(left) != _quantity(right):
        raise ValueError("complex comparison requires the same physical field quantity")


def _subset(cut, indices, *, x=None, x_unit=None):
    return replace(cut, x=np.array(cut.x[indices] if x is None else x, copy=True),
                   power=np.array(cut.power[indices], copy=True),
                   field=np.array(cut.field[indices], copy=True),
                   db=np.array(cut.db[indices], copy=True),
                   x_unit=cut.x_unit if x_unit is None else x_unit,
                   native_indices=_native_indices(cut)[indices],
                   notes=list(cut.notes))


def _difference_db(left, right, field):
    """Preserve the frequency-dependent dBke reference, including at nulls."""
    with np.errstate(divide="ignore", invalid="ignore"):
        levels = 20. * np.log10(np.abs(field))
    if str(left.level_unit).strip().lower() != "dbke":
        return levels
    if left.axis == "frequency":
        frequencies = convert_axis_values(left.x, "frequency", left.x_unit, "Hz")
        with np.errstate(divide="ignore", invalid="ignore"):
            offset = 10. * np.log10(np.where(frequencies > 0, 2. * np.pi * frequencies / C0, np.nan))
        return levels + offset
    fixed = dict(getattr(left, "fixed_coordinates", {}) or {})
    if "frequency" in fixed:
        value, unit = fixed["frequency"]
        frequency_hz = float(convert_axis_values(value, "frequency", unit, "Hz"))
        if np.isfinite(frequency_hz) and frequency_hz > 0:
            return levels + 10. * np.log10(2. * np.pi * frequency_hz / C0)
    # An angular cut has one frequency. Recover its fixed dBke reference from
    # either original cut, avoiding assumptions that dBke is simply 10log(width).
    for original in (left, right):
        if str(original.level_unit).strip().lower() != "dbke":
            continue
        magnitude = np.abs(original.field)
        good = _finite(original.field) & (magnitude > 0) & np.isfinite(original.db)
        if np.any(good):
            offset = float(np.median(original.db[good] - 20. * np.log10(magnitude[good])))
            return levels + offset
    # Without a known frequency or positive dBke reference sample, a nonzero
    # difference has no recoverable log reference. Its native field is still
    # useful; retain -inf only at exact nulls and NaN for unknown log levels.
    return np.where(np.abs(field) == 0, -np.inf, np.nan)


def compare_iq(left, right, *, scaling="native", common_peak=None, floor_db=-60.,
               maximum_gap=None, region=None):
    """A, B, and signed complex A-B at shared physical native coordinates."""
    _compatible(left, right)
    scaling = _scaling_name(scaling)
    ia, ib = aligned_cuts(left, right)
    a = _subset(left, ia)
    b = _subset(right, ib, x=a.x, x_unit=a.x_unit)
    difference = a.field - b.field
    with np.errstate(over="ignore", invalid="ignore"):
        difference_cut = replace(a, field=difference, power=np.abs(difference) ** 2,
                                 db=_difference_db(a, b, difference), label="A - B")
    if scaling == "common_peak" and common_peak is None:
        selected = _selected(a.x, region)
        common_peak = _peak([a.field[selected], b.field[selected]])
    result = {}
    for name, cut in (("left", a), ("right", b), ("difference", difference_cut)):
        row = iq_series(cut, scaling=scaling, common_peak=common_peak, floor_db=floor_db,
                        maximum_gap=maximum_gap, region=region)
        row["runs"] = _runs(row["x"], row["valid"], maximum_gap=maximum_gap,
                            source_indices=(_native_indices(a), _native_indices(b)))
        result[name] = row
    notes = list(dict.fromkeys(list(left.notes) + list(right.notes)))
    notes.append("A - B is a complex field difference; phase-reference differences contribute to it.")
    if left.time_convention != right.time_convention:
        notes.append("Time-convention declarations differ; fields are shown as stored without conjugation.")
    return {**result, "common_count": len(ia), "indices_left": ia, "indices_right": ib,
            "notes": notes}


def compensate_iq(cut, *, phase_offset_deg=0., delay_ns=0., reference_frequency=None,
                   time_sign="+jwt", scaling="native", common_peak=None,
                   floor_db=-60., maximum_gap=None, region=None):
    """Apply explicit display-only reference phase/delay compensation.

    For +jwt multiply by exp(+j 2pi (f-f_ref) delay) and exp(-j phase_offset).
    For -jwt reverse the delay sign. The phase offset is the geometric phase
    at f_ref in either convention. Delay does not estimate target range.
    """
    if time_sign not in {"+jwt", "-jwt"}:
        raise ValueError("choose +jwt or -jwt for delay compensation")
    if not np.isfinite(phase_offset_deg) or not np.isfinite(delay_ns):
        raise ValueError("phase offset and delay must be finite")
    if delay_ns != 0 and cut.axis != "frequency":
        raise ValueError("delay compensation requires a frequency sweep")
    x = np.asarray(cut.x, dtype=float)
    rotation = np.full(x.shape, -np.deg2rad(float(phase_offset_deg)))
    if cut.axis == "frequency":
        finite_x = x[np.isfinite(x)]
        if reference_frequency is None:
            reference_frequency = float((np.min(finite_x) + np.max(finite_x)) / 2.) if len(finite_x) else 0.
        if not np.isfinite(reference_frequency):
            raise ValueError("reference frequency must be finite")
        relative_hz = convert_axis_values(x - reference_frequency, "frequency", cut.x_unit, "Hz")
        rotation += (1 if time_sign == "+jwt" else -1) * 2. * np.pi * relative_hz * float(delay_ns) * 1.e-9
    else:
        reference_frequency = None
    corrected = np.asarray(cut.field, dtype=complex) * np.exp(1j * rotation)
    corrected_cut = replace(cut, field=corrected, label=f"{cut.label} | compensated", notes=list(cut.notes))
    if _scaling_name(scaling) == "common_peak" and common_peak is None:
        common_peak = _peak([np.asarray(cut.field)[_selected(x, region)]])
    kwargs = dict(scaling=scaling, common_peak=common_peak, floor_db=floor_db,
                  maximum_gap=maximum_gap, region=region)
    return {"original": iq_series(cut, **kwargs), "compensated": iq_series(corrected_cut, **kwargs),
            "reference_frequency": reference_frequency, "phase_offset_deg": float(phase_offset_deg),
            "delay_ns": float(delay_ns), "time_sign": time_sign,
            "notes": ["Display-only phase/delay compensation; the stored dataset is unchanged."]}


def repeat_iq(cuts, coordinate, *, scaling="native", common_peak=None, floor_db=-60.):
    """One sample per run at the same physical coordinate, without interpolation.

    The caller supplies cuts at the same fixed nonsweep coordinates and channel.
    The requested sweep coordinate snaps once to A; other runs must have that
    physical native sample. Population statistics do not mix sweep coordinates.
    """
    cuts = list(cuts)
    if not cuts:
        raise ValueError("select at least one cut for repeatability")
    if not np.isfinite(coordinate):
        raise ValueError("coordinate must be finite")
    requested_coordinate = float(coordinate)
    reference = cuts[0]
    finite = np.flatnonzero(np.isfinite(reference.x))
    if not len(finite):
        raise ValueError("reference cut has no finite coordinates")
    reference_index = int(finite[np.argmin(np.abs(reference.x[finite] - coordinate))])
    coordinate = float(reference.x[reference_index])
    fields, levels, labels, indices, positions, cut_indices, skipped, phases, phase_valid = [], [], [], [], [], [], [], [], []
    metadata = []
    for cut_index, cut in enumerate(cuts):
        try:
            _compatible(reference, cut)
            ia, ib = aligned_cuts(reference, cut)
        except ValueError as exc:
            skipped.append({"cut_index": cut_index, "label": cut.label, "reason": str(exc)})
            continue
        matches = np.flatnonzero(ia == reference_index)
        if not len(matches):
            skipped.append({"cut_index": cut_index, "label": cut.label, "reason": "No matching native coordinate."})
            continue
        index = int(ib[matches[0]])
        value = complex(cut.field[index])
        if not (np.isfinite(value.real) and np.isfinite(value.imag)):
            skipped.append({"cut_index": cut_index, "label": cut.label, "reason": "Field is missing at this coordinate."})
            continue
        phase = phase_trace(cut.x, cut.field, floor_db=floor_db)
        fields.append(value)
        levels.append(float(cut.db[index]))
        labels.append(cut.label)
        indices.append(int(_native_indices(cut)[index]))
        positions.append(index)
        cut_indices.append(cut_index)
        metadata.append({"linear_quantity": _quantity(cut), "level_unit": cut.level_unit,
                         "field_unit": field_unit(cut),
                         "polarization": str(getattr(cut, "polarization", "") or ""),
                         "fixed_coordinates": dict(getattr(cut, "fixed_coordinates", {}) or {}),
                         "notes": list(cut.notes)})
        phases.append(float(np.rad2deg(phase["wrapped"][index])))
        phase_valid.append(bool(phase["valid"][index]))
    native = np.asarray(fields, dtype=complex)
    scaling = _scaling_name(scaling)
    display, scale = _display(native, scaling, common_peak)
    valid = _finite(display)
    mean = complex(np.mean(native)) if len(native) else complex(np.nan, np.nan)
    rms = float(np.sqrt(np.mean(np.abs(native - mean) ** 2))) if len(native) else np.nan
    display_mean = complex(np.mean(display[valid])) if np.any(valid) else complex(np.nan, np.nan)
    display_rms = float(np.sqrt(np.mean(np.abs(display[valid] - display_mean) ** 2))) if np.any(valid) else np.nan
    return {"coordinate": coordinate, "requested_coordinate": requested_coordinate,
            "reference_index": reference_index, "x_unit": reference.x_unit, "axis": reference.axis,
            "reference_native_index": int(_native_indices(reference)[reference_index]),
            "field": display, "native_field": native, "db": np.asarray(levels),
            "phase_deg": np.asarray(phases), "phase_valid": np.asarray(phase_valid, dtype=bool),
            "valid": valid, "labels": labels, "indices": np.asarray(indices, dtype=int),
            "positions": np.asarray(positions, dtype=int),
            "metadata": metadata,
            "cut_indices": np.asarray(cut_indices, dtype=int), "mean_field": mean,
            "rms_distance": rms, "display_mean_field": display_mean,
            "display_rms_distance": display_rms, "count": len(native), "skipped": skipped,
            "field_unit": field_unit(reference), "level_unit": reference.level_unit,
            "linear_quantity": _quantity(reference),
            "display_unit": field_unit(reference) if scaling == "native" else "dimensionless",
            "scaling": scaling, "scale": scale,
            "notes": ["One native sample per selected run at fixed frequency, angle, and polarization.",
                      "RMS distance is population spread about the complex mean; it is not a noise or coherence estimate."]}
