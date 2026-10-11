"""Physical image records, native cuts, and explicit A/B comparison policies.

Arrays use Matplotlib's row/column order (y, x). Source arrays are never edited.
An image record can reference an artist's native snapshot; dialogs freeze their
own temporary-file snapshots so later plots or dataset edits cannot change A/B.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.isar.comparison import comparison_issues, roi_statistics
from GRIM_Backend.datasets.constants import C0

AXES = ("azimuth", "elevation", "frequency")
AXIS_ATTRIBUTES = {"azimuth": "azimuths", "elevation": "elevations", "frequency": "frequencies"}
LENGTH_SCALE = {"m": 1., "cm": .01, "mm": .001, "in": .0254, "ft": .3048}


def compact_metadata(mapping):
    """Copy only small comparison declarations, never solver arrays/meshes."""
    keys = ("phase_reference", "polarization_basis", "amplitude_convention", "complex_field_domain",
            "phase_center_xyz_m", "time_convention", "source_file", "filename", "source", "name",
            "azimuth", "elevation", "frequency", "angular_coordinate_system", "rcs_linear_quantity", "rcs_log_unit")
    result = {}
    for key in keys:
        value = (mapping or {}).get(key)
        if value is None or isinstance(value, (str, bool, int, float)):
            if key in (mapping or {}):
                result[key] = value
        elif key == "phase_center_xyz_m" and isinstance(value, (tuple, list, np.ndarray)) and np.shape(value) == (3,):
            result[key] = np.asarray(value, float).tolist()
    return result


@dataclass
class ImageRecord:
    name: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    x_axis: str
    y_axis: str
    x_unit: str
    y_unit: str
    quantity: str
    value_unit: str
    logarithmic: bool = False
    phase: bool = False
    notes: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    storage: object = None

    def frozen(self):
        storage = common.NativeSamples.image(np.ma.filled(self.values, np.nan))
        return replace(self, x=np.array(self.x, copy=True), y=np.array(self.y, copy=True),
                       values=storage.values, metadata=deepcopy(self.metadata), storage=storage)

    def close(self):
        if self.storage is not None:
            self.values = np.empty((0,0))
            self.storage.close()
            self.storage = None


def heatmap_plane(dataset, indices, x_axis, y_axis, *, remaining_index=None, phase=False):
    """Selected X-by-Y plane, or mean-power/circular-mean over the third axis.

    ``indices`` is in azimuth/elevation/frequency/polarization order. The
    polarization must be a singleton. Phase averaging uses unit phasors and
    returns a complex unit value; magnitude averaging is in linear power.
    """
    if x_axis not in AXES or y_axis not in AXES or x_axis == y_axis:
        raise ValueError("Choose two different waterfall axes")
    remaining = next(a for a in AXES if a not in (x_axis, y_axis))
    selected = [np.asarray(ids, dtype=int) for ids in indices]
    axis_index = AXES.index(remaining)
    if remaining_index is not None:
        selected[axis_index] = np.asarray([remaining_index], dtype=int)
    selection = np.ix_(*selected)
    data = dataset.rcs_slice(selection)[..., 0] if phase else dataset.rcs_power[selection][..., 0]
    if phase:
        finite = np.isfinite(data) & (np.abs(data)>0)
        data = np.where(finite, np.exp(1j * np.angle(data)), np.nan + 1j * np.nan)
    else:
        finite = np.isfinite(data) & (data >= 0)
    count = finite.sum(axis=axis_index)
    total = np.where(finite, data, 0).sum(axis=axis_index)
    with np.errstate(invalid="ignore", divide="ignore"):
        plane = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if phase:
        plane = np.where(np.abs(plane) > 1.e-12, plane, np.nan + 1j * np.nan)
    remaining_order = [a for a in AXES if a != remaining]
    if remaining_order[0] != x_axis:
        plane = plane.T
    frequencies = np.asarray(dataset.frequencies[selected[2]], float)
    if "frequency" == x_axis:
        frequencies = frequencies[:, None]
    elif "frequency" == y_axis:
        frequencies = frequencies[None, :]
    else:
        # For frequency aggregation the renderer converts every frequency to
        # the requested physical ordinate before averaging (see Heatmap).
        frequencies = frequencies[0] if len(frequencies) == 1 else frequencies
    return plane, frequencies


def heatmap_display_plane(dataset, indices, x_axis, y_axis, *, remaining_index=None, phase=False, linear=False):
    """Shared GUI/headless X-by-Y display plane with physical aggregation."""
    raw, frequency = heatmap_plane(dataset,indices,x_axis,y_axis,remaining_index=remaining_index,phase=phase)
    if phase:
        degrees = np.rad2deg(np.angle(raw))
        return np.mod(degrees,360.) if (dataset.units or {}).get("phase_wrap") == "0_360" else (degrees+180.)%360.-180.
    if linear:
        return raw
    if remaining_index is None and "frequency" not in (x_axis,y_axis):
        total, count = np.zeros(raw.shape), np.zeros(raw.shape,dtype=np.int64)
        for fi in indices[2]:
            plane, f = heatmap_plane(dataset,indices,x_axis,y_axis,remaining_index=fi)
            valid = np.isfinite(plane) & (plane >= 0)
            # Convert a unit power to the requested log reference to retain
            # legitimate zero power in the mean without inventing a dB floor.
            factor = 10.**(dataset.linear_to_default_db(1.,frequency_value=f,eps=0.)/10.)
            total += np.where(valid,plane*factor,0.)
            count += valid
        with np.errstate(invalid="ignore",divide="ignore"):
            mean = np.where(count,total/np.maximum(count,1),np.nan)
            return np.where(mean>0,10*np.log10(mean),np.nan)
    return dataset.linear_to_default_db(np.where(np.isfinite(raw)&(raw>0),raw,np.nan),frequency_value=frequency,eps=0.)


def attach_isar_record(owner, mesh, band, params):
    """Attach frame/recipe provenance to a rendered ISAR image for A/B capture."""
    dataset = params["dataset"]
    reduced = bool(band.get("display_decimated"))
    if reduced:
        x = (np.asarray(band["display_x_edges"][:-1]) + band["display_x_edges"][1:]) / 2
        y = (np.asarray(band["display_y_edges"][:-1]) + band["display_y_edges"][1:]) / 2
    else:
        x, y = band["x_range"], band["y_range"]
    source = {"units": compact_metadata(dataset.units), "metadata": compact_metadata(dataset.extra),
              "selected_polarization": str(dataset.polarizations[params["pol_idx"]]),
              "selected_elevation_degrees": float(params["elevation_deg"]),
              "selected_frequency_values_hz": np.asarray(dataset._frequency_value_to_hz(
                  dataset.frequencies[np.asarray(params.get("freq_indices_sorted", []),dtype=int)])).tolist(),
              "selected_azimuth_values_native": common.convert_axis_values(
                  band["az_values"], "azimuth", "deg", common.axis_unit(dataset, "azimuth")).tolist()}
    manifest = {"source": source, "formation": {"window": params.get("window_name"),
                "reconstruction": params.get("recon")}}
    contract = {key: deepcopy(band.get(key)) for key in
                ("image_contract", "resolved_reconstruction", "az_values")}
    notes = ("Overview block maxima: cuts, ROI and differences use displayed blocks, not native pixels.",) if reduced else ()
    if not owner._plot_scale_is_linear():
        notes += ("ISAR display intensity uses the formation display floor of −120 dB.",)
    names = getattr(owner,"_selected_datasets",lambda:[])()
    source_name = next((name for name,candidate in names if candidate is dataset),"ISAR")
    mesh._grim_image_record = ImageRecord(
        f"{source_name} | ISAR | {source['selected_polarization']} | {band['az_values'][0]:g}–{band['az_values'][-1]:g}°",
        np.asarray(x), np.asarray(y), np.asanyarray(mesh.get_array()).reshape(len(y),len(x)), "cross_range", "range",
        params["unit_name"], params["unit_name"], "image_intensity", "linear" if owner._plot_scale_is_linear() else "dB",
        logarithmic=not owner._plot_scale_is_linear(), notes=notes,
        metadata={"kind": "isar", "source_name":source_name, "manifest": manifest, "band": contract})


def records_from_figure(owner):
    return [artist._grim_image_record for ax in owner.plot_figure.axes
            for artist in (*ax.images, *ax.collections)
            if getattr(artist, "_grim_image_record", None) is not None]


def ordered(record):
    x, y = np.asarray(record.x, float), np.asarray(record.y, float)
    values = np.ma.filled(record.values, np.nan)
    if x.ndim != 1 or y.ndim != 1 or values.shape != (len(y), len(x)):
        raise ValueError("Image dimensions do not match its coordinate axes")
    if not len(x) or not len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Image axes must contain finite coordinates")
    xi, yi = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
    if np.any(np.diff(x[xi]) <= 0) or np.any(np.diff(y[yi]) <= 0):
        raise ValueError("Image coordinates must be unique for linked inspection")
    return x[xi], y[yi], np.asarray(values)[np.ix_(yi, xi)]


def cuts(record, x, y):
    """Nearest native row/column and actual selected center coordinates."""
    xs, ys = np.asarray(record.x), np.asarray(record.y)
    ix, iy = int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))
    return xs, np.asarray(record.values[iy, :]), ys, np.asarray(record.values[:, ix]), float(xs[ix]), float(ys[iy])


def _convert_axis(values, axis, source_unit, target_unit):
    if source_unit == target_unit:
        return np.asarray(values)
    if axis in AXES:
        return common.convert_axis_values(values, axis, source_unit, target_unit)
    if source_unit in LENGTH_SCALE and target_unit in LENGTH_SCALE:
        return np.asarray(values) * (LENGTH_SCALE[source_unit] / LENGTH_SCALE[target_unit])
    raise ValueError(f"Cannot convert {source_unit} to {target_unit}")


def comparison_advisories(a, b):
    notes = list(a.notes) + list(b.notes)
    if (a.x_axis, a.y_axis) != (b.x_axis, b.y_axis):
        notes.append("Axis meanings differ; compare as supplied uses numeric coordinates only.")
    if a.quantity != b.quantity or a.value_unit != b.value_unit:
        notes.append("Image ordinate quantities or units differ; select relative-peak or explicit numeric comparison.")
    if a.metadata.get("kind") == b.metadata.get("kind") == "isar":
        notes.extend(comparison_issues(a.metadata["manifest"], a.metadata["band"], b.metadata["manifest"], b.metadata["band"]))
    for key in ("polarization", "remaining_selection", "coordinate_system", "frequency_values_hz", "peak_normalized"):
        if a.metadata.get(key) != b.metadata.get(key):
            notes.append(f"Different {key.replace('_', ' ')}; images are compared as supplied.")
    return tuple(dict.fromkeys(notes))


def _log_reference_factor(record, x=None, y=None):
    if record.value_unit.lower() != "dbke" or record.metadata.get("frequency_aggregation"):
        return 1.
    if record.x_axis == "frequency":
        frequency = common.convert_axis_values(record.x if x is None else x,"frequency",record.x_unit,"Hz")[None,:]
    elif record.y_axis == "frequency":
        frequency = common.convert_axis_values(record.y if y is None else y,"frequency",record.y_unit,"Hz")[:,None]
    else:
        selection = record.metadata.get("remaining_selection")
        if not selection or selection[0] != "frequency":
            raise ValueError("dBke comparison needs the source frequency reference; choose displayed numeric comparison")
        frequency = common.convert_axis_values([selection[1]],"frequency",selection[2],"Hz")[0]
    return C0/(2*np.pi*frequency)


def _to_linear(values, record, *, physical=False, x=None, y=None):
    if not record.logarithmic:
        return np.asarray(values)
    result = np.power(10.,np.asarray(values)/10.)
    return result * _log_reference_factor(record,x,y) if physical else result


def _sample_grid(xb, yb, values, x, y, policy):
    """Sample only inside B support; interpolate linear values, not dB.

    An exact center is copied independently of missing neighbors. Bilinear
    samples require only corners with nonzero interpolation weights.
    """
    def positions(source, target):
        pos = np.searchsorted(source, target)
        lo, hi = np.clip(pos - 1, 0, len(source)-1), np.clip(pos, 0, len(source)-1)
        span = source[hi] - source[lo]
        weight = np.divide(target-source[lo], span, out=np.zeros_like(target), where=span != 0)
        near_lo = np.isclose(target, source[lo], rtol=0, atol=8*np.finfo(float).eps*max(1., np.max(abs(source))))
        near_hi = np.isclose(target, source[hi], rtol=0, atol=8*np.finfo(float).eps*max(1., np.max(abs(source))))
        weight[near_lo], weight[near_hi] = 0., 1.
        return lo, hi, weight
    xl, xh, wx = positions(xb, x)
    yl, yh, wy = positions(yb, y)
    if policy == "nearest":
        return values[np.ix_(np.where(wy <= .5, yl, yh), np.where(wx <= .5, xl, xh))]
    out = np.zeros((len(y), len(x)), dtype=np.result_type(values.dtype, float))
    for rows, rowweight in ((yl, 1-wy), (yh, wy)):
        for cols, colweight in ((xl, 1-wx), (xh, wx)):
            weight = rowweight[:, None] * colweight[None, :]
            sample = values[np.ix_(rows, cols)]
            out += np.where(weight > 0, sample * weight, 0)
    return out


def compare_records(a, b, *, policy="exact", mode="physical", as_supplied=False, maximum_cells=None):
    """A minus B on A's overlapping grid with explicit interpolation policy.

    Physical logarithmic differences are ratios in dB. Phase differences are
    wrapped to [-180,180). Relative-peak differences independently normalize
    images to their positive peak and carry no absolute calibration claim.
    """
    if policy not in ("exact", "nearest", "linear") or mode not in ("physical", "relative", "displayed"):
        raise ValueError("Unknown image comparison policy")
    if mode == "relative" and (a.phase or b.phase):
        raise ValueError("Relative-peak normalization needs two level images; phase has no intensity peak")
    xa, ya, av = ordered(a)
    xb, yb, bv = ordered(b)
    source_xb,source_yb = xb,yb
    coordinate_notes = ()
    axis_match = (a.x_axis, a.y_axis) == (b.x_axis, b.y_axis)
    if not axis_match and not as_supplied:
        raise ValueError("Axis meanings differ. Enable compare as supplied to use numeric coordinates.")
    if axis_match:
        try:
            converted_x = _convert_axis(xb, a.x_axis, b.x_unit, a.x_unit)
            converted_y = _convert_axis(yb, a.y_axis, b.y_unit, a.y_unit)
            xb,yb = converted_x,converted_y
        except ValueError:
            if not as_supplied:
                raise
            coordinate_notes = ("Axis units could not be converted; explicit compare-as-supplied uses numeric coordinates without a physical unit conversion.",)
    if mode == "physical" and (a.quantity != b.quantity or a.phase != b.phase):
        raise ValueError("Physical quantities differ. Choose relative-peak or explicit displayed numeric comparison.")
    if mode == "physical" and a.value_unit != b.value_unit and a.logarithmic == b.logarithmic and not (a.quantity == b.quantity and a.quantity in ("image_intensity","range_profile_intensity")):
        raise ValueError("Ordinate units differ. Choose relative-peak or explicit displayed numeric comparison.")
    xtol = 8*np.finfo(float).eps*max(1.,np.max(abs(xa)),np.max(abs(xb)))
    ytol = 8*np.finfo(float).eps*max(1.,np.max(abs(ya)),np.max(abs(yb)))
    ix = np.flatnonzero((xa >= xb[0]-xtol) & (xa <= xb[-1]+xtol))
    iy = np.flatnonzero((ya >= yb[0]-ytol) & (ya <= yb[-1]+ytol))
    if not len(ix) or not len(iy):
        raise ValueError("Images have no overlapping sample centers")
    if maximum_cells is not None and len(ix)*len(iy) > maximum_cells:
        raise ValueError("The overlap exceeds the requested cell budget; select smaller input images or raise maximum_cells")
    x, y = xa[ix], ya[iy]
    left = av[np.ix_(iy, ix)]
    if mode == "displayed":
        values_b = bv
        unit = f"numeric A ({a.value_unit}) − B ({b.value_unit})"
    elif a.phase:
        if mode == "relative":
            raise ValueError("Relative-peak normalization is a level comparison; use wrapped phase difference")
        values_b = np.exp(1j * np.deg2rad(bv))
        unit = "wrapped phase difference (deg)"
    else:
        values_b = _to_linear(bv, b, physical=mode=="physical",x=source_xb,y=source_yb)
        left = _to_linear(av, a, physical=mode=="physical",x=xa,y=ya)[np.ix_(iy,ix)]
        unit = "relative-peak difference (dB)" if mode == "relative" else ("difference (dB)" if a.logarithmic or b.logarithmic else a.value_unit)
        if mode == "relative":
            pa = np.nanmax(_to_linear(av, a))
            pb = np.nanmax(values_b)
            if not (np.isfinite(pa) and np.isfinite(pb) and pa > 0 and pb > 0):
                raise ValueError("Relative-peak comparison needs a positive finite peak in each image")
            left, values_b = left/pa, values_b/pb
    if policy == "exact":
        def match(source, target):
            tolerance = 8*np.finfo(float).eps*max(1., np.max(abs(source)), np.max(abs(target)))
            ids = common.unique_axis_selection(source, target, tolerance)
            if ids is None:
                raise ValueError("Grids differ. Select nearest-neighbor or linear interpolation on the overlapping grid.")
            return ids
        right = values_b[np.ix_(match(yb, y), match(xb, x))]
    else:
        right = _sample_grid(xb, yb, values_b, x, y, policy)
    with np.errstate(invalid="ignore", divide="ignore"):
        if a.phase and mode != "displayed":
            right = np.where(abs(right) > 1.e-12, np.rad2deg(np.angle(right)), np.nan)
            delta = (left - right + 180.) % 360. - 180.
        elif mode != "displayed" and (mode == "relative" or a.logarithmic or b.logarithmic):
            reference = _log_reference_factor(a,x,y) if a.logarithmic and mode == "physical" else 1.
            left = np.where(left > 0, 10*np.log10(left/reference), np.nan)
            right = np.where(right > 0, 10*np.log10(right/reference), np.nan)
            delta = left - right
        else:
            delta = left - right
    level_unit = (a.value_unit if a.logarithmic else ("dB relative to one " + a.value_unit)) if a.logarithmic or b.logarithmic else a.value_unit
    return {"x": x, "y": y, "a": left, "b": right, "difference": delta, "unit": unit,"level_unit":level_unit,
            "phase":bool(a.phase and mode!="displayed"),
            "notes": comparison_advisories(a, b)+coordinate_notes, "policy": policy}


def summarize_roi(result, bounds=None):
    x, y = result["x"], result["y"]
    if bounds is None:
        bounds = (min(x), max(x), min(y), max(y))
    x0, x1, y0, y1 = bounds
    selection = np.ix_((y >= min(y0,y1)) & (y <= max(y0,y1)), (x >= min(x0,x1)) & (x <= max(x0,x1)))
    delta = result["difference"][selection]
    stats = roi_statistics(delta)
    if result.get("phase"):
        finite = delta[np.isfinite(delta)]
        resultant = np.mean(np.exp(1j*np.deg2rad(finite)))
        stats["mean_delta_db"] = float(np.rad2deg(np.angle(resultant))) if abs(resultant)>1.e-12 else float("nan")
    # Reuse the existing numerical summary but make units generic.
    return {key.removesuffix("_db"): value for key, value in stats.items()} | {
        "selected_cells": int(delta.size), "excluded_cells": int(delta.size - np.isfinite(delta).sum()),
        "unit": result["unit"], "mean_definition":"circular" if result.get("phase") else "arithmetic"}
